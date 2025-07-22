#include <cuda.h>
#include <nvm_ctrl.h>
#include <nvm_types.h>
#include <nvm_queue.h>
#include <nvm_util.h>
#include <nvm_admin.h>
#include <nvm_error.h>
#include <nvm_cmd.h>
#include <string>
#include <stdexcept>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <ctrl.h>
#include <buffer.h>
#include "settings.h"
#include <event.h>
#include <queue.h>
#include <nvm_parallel_queue.h>
#include <nvm_io.h>
#include <page_cache.h>
#include <util.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <algorithm>
#include <iostream>
#include <list>
#include <numeric>
#include <random>
#include <vector>
#include <map>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#ifdef __DIS_CLUSTER__
#include <sisci_api.h>
#endif

using error = std::runtime_error;
using std::string;



//uint32_t n_ctrls = 1;
const char* const ctrls_paths[] = {"/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm2", "/dev/libnvm3", "/dev/libnvm4", "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7"};

__global__
void flush_kernel(page_cache_d_t* cache) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t page = tid;
    // if (tid == 0) {
    //     hexdump(cache->base_addr, 4096);
    // }
    if (page < cache->n_pages) {
        uint32_t v = cache->cache_pages[page].page_take_lock.load(simt::memory_order_acquire);
        if (v != FREE) {
            uint32_t previous_global_address = cache->cache_pages[page].page_translation;
            uint32_t previous_range = previous_global_address & cache->n_ranges_mask;
            uint32_t previous_address = previous_global_address >> cache->n_ranges_bits;
            uint32_t expected_state = cache->ranges[previous_range][previous_address].state.load(simt::memory_order_acquire);
            if (expected_state == VALID_DIRTY) {
                uint64_t ctrl = get_backing_ctrl_(previous_address, cache->n_ctrls, cache->ranges_dists[previous_range]);
                //uint64_t get_backing_page(const uint64_t page_start, const size_t page_offset, const uint64_t n_ctrls, const data_dist_t dist) {
                uint64_t index = get_backing_page_(cache->ranges_page_starts[previous_range], previous_address,
                                                   cache->n_ctrls, cache->ranges_dists[previous_range]);
                // printf("Eviciting range_id: %llu\tpage_id: %llu\tctrl: %llx\tindex: %llu\n",
                //        (unsigned long long) previous_range, (unsigned long long)previous_address,
                //        (unsigned long long) ctrl, (unsigned long long) index);
                if (ctrl == ALL_CTRLS) {
                    for (ctrl = 0; ctrl < cache->n_ctrls; ctrl++) {
                        Controller* c = cache->d_ctrls[ctrl];
                        uint32_t queue = (tid/32) % (c->n_qps);
                        write_data(cache, (c->d_qps)+queue, (index*cache->n_blocks_per_page), cache->n_blocks_per_page, page);
                    }
                }
                else {

                    Controller* c = cache->d_ctrls[ctrl];
                    uint32_t queue = (tid/32) % (c->n_qps);

                    //index = ranges_page_starts[previous_range] + previous_address;


                    write_data(cache, (c->d_qps)+queue, (index*cache->n_blocks_per_page), cache->n_blocks_per_page, page);
                }
            }
        }

    }
}
__global__
void sequential_access_kernel(array_d_t<unsigned long long int>* dr, uint64_t n_reqs, unsigned long long* req_count, uint64_t reqs_per_thread) {

    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_reqs) {
        for (size_t i = 0; i < reqs_per_thread; i++)
            req_count += (*dr)[(tid)];

    }

}

__global__
void random_access_kernel(array_d_t<unsigned long long int>* dr, uint64_t n_reqs, unsigned long long* req_count, uint64_t* assignment, uint64_t reqs_per_thread, unsigned long long int* f_in) {

    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_reqs) {
        for (size_t i = 0; i < reqs_per_thread; i++) {
            uint64_t idx = assignment[tid];
            unsigned long long int v = f_in[idx];
            dr->AtomicAdd(idx, v);
        }


    }

}

int main(int argc, char** argv) {

    Settings settings;
    try
    {
        settings.parseArguments(argc, argv);
    }
    catch (const string& e)
    {
        fprintf(stderr, "%s\n", e.c_str());
        fprintf(stderr, "%s\n", Settings::usageString(argv[0]).c_str());
        return 1;
    }


    cudaDeviceProp properties;
    if (cudaGetDeviceProperties(&properties, settings.cudaDevice) != cudaSuccess)
    {
        fprintf(stderr, "Failed to get CUDA device properties\n");
        return 1;
    }

    try {
        uint64_t b_size = settings.blkSize;//64;
        uint64_t g_size = (settings.numThreads + b_size - 1)/b_size;//80*16;
        uint64_t n_threads = b_size * g_size;


        uint64_t page_size = settings.pageSize;
        uint64_t n_pages = settings.numPages;
        uint64_t total_cache_size = (page_size * n_pages);
        #define TYPE unsigned long long int
        uint64_t n_elems = settings.numThreads;
        uint64_t t_size = n_elems * sizeof(TYPE);
        const char* input_f;

        if(settings.input == nullptr){
            fprintf(stderr, "Input file required\n");
            return 1;
        }
        else {
            input_f = settings.input;
            printf("File is : %s\n",input_f);
        }
        //Controller ctrl(settings.controllerPath, settings.nvmNamespace, settings.cudaDevice);
        void* map_in;
        int fd_in;
        struct stat sb_in;

        if((fd_in = open(input_f, O_RDWR)) == -1){
            fprintf(stderr, "Input file cannot be opened\n");
            return 1;
        }

        fstat(fd_in, &sb_in);

        map_in = mmap(NULL, t_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_LOCKED, fd_in, 0);

        if((map_in == (void*)-1)){
                fprintf(stderr,"Input file map failed %d\n",map_in);
                return 1;
        }

        cuda_err_chk(cudaSetDevice(settings.cudaDevice));

        unsigned long long int* f_in_d;
        cuda_err_chk(cudaHostRegister(map_in, t_size, cudaHostRegisterDefault));
        cuda_err_chk(cudaHostGetDevicePointer(&f_in_d, map_in, 0));

        std::vector<Controller*> ctrls(settings.n_ctrls);
        for (size_t i = 0 ; i < settings.n_ctrls; i++)
            ctrls[i] = new Controller(ctrls_paths[i], settings.nvmNamespace, settings.cudaDevice, settings.queueDepth, settings.numQueues);
        return 1;

        //auto dma = createDma(ctrl.ctrl, NVM_PAGE_ALIGN(64*1024*10, 1UL << 16), settings.cudaDevice, settings.adapter, settings.segmentId);

        //std::cout << dma.get()->vaddr << std::endl;
        //QueuePair h_qp(ctrl, settings, 1);
        //std::cout << "in main: " << std::hex << h_qp.sq.cid << "raw: " << h_qp.sq.cid<< std::endl;
        //std::memset(&h_qp, 0, sizeof(QueuePair));
        //prepareQueuePair(h_qp, ctrl, settings, 1);
        //const uint32_t ps, const uint64_t np, const uint64_t c_ps, const Settings& settings, const Controller& ctrl)
        //
        /*
        Controller** d_ctrls;
        cuda_err_chk(cudaMalloc(&d_ctrls, n_ctrls*sizeof(Controller*)));
        for (size_t i = 0; i < n_ctrls; i++)
            cuda_err_chk(cudaMemcpy(d_ctrls+i, &(ctrls[i]->d_ctrl), sizeof(Controller*), cudaMemcpyHostToDevice));
        */

        //uint64_t n_pages = total_cache_size/page_size;


        page_cache_t h_pc(page_size, n_pages, settings.cudaDevice, ctrls[0][0], (uint64_t) 64, ctrls);
        std::cout << "finished creating cache\n";

        //QueuePair* d_qp;
        page_cache_d_t* d_pc = (h_pc.d_pc_ptr);


        range_t<unsigned long long int> h_range((uint64_t)0, (uint64_t)n_elems, (uint64_t)0, (uint64_t)((t_size+page_size-1)/page_size), (uint64_t)0, (uint64_t)page_size, &h_pc, settings.cudaDevice);
        range_t<unsigned long long int>* d_range = (range_t<unsigned long long int>*) h_range.d_range_ptr;

        std::vector<range_t<unsigned long long int>*> vr(1);
        vr[0] = & h_range;
        //(const uint64_t num_elems, const uint64_t disk_start_offset, const std::vector<range_t<T>*>& ranges, Settings& settings)
        array_t<unsigned long long int> a(n_elems, 0, vr, settings.cudaDevice);


        std::cout << "finished creating range\n";




        unsigned long long* d_req_count;
        cuda_err_chk(cudaMalloc(&d_req_count, sizeof(unsigned long long)));
        cuda_err_chk(cudaMemset(d_req_count, 0, sizeof(unsigned long long)));
        std::cout << "atlaunch kernel\n";
        char st[15];
        cuda_err_chk(cudaDeviceGetPCIBusId(st, 15, settings.cudaDevice));
        std::cout << st << std::endl;
        std::cout << std::dec << "File Size: " << t_size << " Num Elems In File: " <<
            (t_size/sizeof(TYPE)) << " Num elems in vec: " << n_threads << std::endl;
        //uint64_t* assignment;
        uint64_t* d_assignment;
        if (settings.random) {
            std::vector<uint64_t> assignment(n_threads);
            uint64_t start = 0;
            std::iota(assignment.begin(), assignment.end(), start);
            std::cout << "Finished iota\n";
            // for (size_t i = 0; i < assignment.size(); i++) {
            //     assignment[i] = start++;
            // }
            std::cout <<std::dec << "last vec elem i : " << assignment.size()-1 << " val: " << assignment[assignment.size()-1] << std::endl;
            // for (size_t i = 0; i < assignment.size(); i++) {
            //     if (assignment[i] >= n_threads)
            //         std::cout <<std::dec << "i : " << i << " val: " << assignment[i] << std::endl;
            // }
            std::cout << std::dec << "max elem: " << *std::max_element(assignment.begin(), assignment.end()) << std::endl;
            std::shuffle(assignment.begin(), assignment.end(), std::mt19937_64{std::random_device{}()});
            std::cout << "Finished shuffle\n";

            cuda_err_chk(cudaMallocManaged(&d_assignment, n_threads*sizeof(uint64_t)));
            cuda_err_chk(cudaMemcpy(d_assignment, assignment.data(),  n_threads*sizeof(uint64_t), cudaMemcpyHostToDevice));
        }
        //return 1;
        Event before;
        //access_kernel<<<g_size, b_size>>>(h_pc.d_ctrls, d_pc, page_size, n_threads, d_req_count, settings.n_ctrls, d_assignment, settings.numReqs);
        if (settings.random)
            random_access_kernel<<<g_size, b_size>>>(a.d_array_ptr, n_threads, d_req_count, d_assignment, settings.numReqs, f_in_d);
        cuda_err_chk(cudaDeviceSynchronize());
        std::cout << "Finished Random access kernel\n";

        flush_kernel<<<n_pages, 1>>>(d_pc);

        //new_kernel<<<1,1>>>();
        //uint8_t* ret_array = (uint8_t*) malloc(n_pages*page_size);

        //cuda_err_chk(cudaMemcpy(ret_array, h_pc.base_addr,page_size*n_pages, cudaMemcpyDeviceToHost));
        cuda_err_chk(cudaDeviceSynchronize());


        double elapsed = 0;
        uint64_t ios = g_size*b_size*settings.numReqs;
        uint64_t data = ios*sizeof(uint64_t);
        double iops = ((double)ios)/(elapsed/1000000);
        double bandwidth = (((double)data)/(elapsed/1000000))/(1024ULL*1024ULL*1024ULL);
        a.print_reset_stats();
        std::cout << std::dec << "Elapsed Time: " << elapsed << "\tNumber of Read Ops: "<< ios << "\tData Size (bytes): " << data << std::endl;
        std::cout << std::dec << "Read Ops/sec: " << iops << "\tEffective Bandwidth(GB/S): " << bandwidth << std::endl;

        //std::cout << std::dec << ctrls[0]->ns.lba_data_size << std::endl;

        //std::ofstream ofile("../data", std::ios::binary | std::ios::trunc);
        //ofile.write((char*)ret_array, data);
        //ofile.close();

        for (size_t i = 0 ; i < settings.n_ctrls; i++)
            delete ctrls[i];
        //hexdump(ret_array, n_pages*page_size);
/*
        cudaFree(d_qp);
        cudaFree(d_pc);
        cudaFree(d_req_count);
        free(ret_array);
*/

        //std::cout << "END\n";

        //std::cout << RAND_MAX << std::endl;

    }
    catch (const error& e) {
        fprintf(stderr, "Unexpected error: %s\n", e.what());
        return 1;
    }



}
