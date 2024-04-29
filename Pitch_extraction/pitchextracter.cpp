#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <future>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <essentia/algorithmfactory.h>
#include <essentia/essentiamath.h>
#include <essentia/pool.h>
#include <filesystem>

namespace fs = std::filesystem;
using namespace essentia;
using namespace essentia::standard;

class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
    
public:
    ThreadPool(size_t);
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;
    ~ThreadPool();
};

ThreadPool::ThreadPool(size_t threads) : stop(false) {
    for(size_t i = 0;i<threads;++i)
        workers.emplace_back(
            [this] {
                for(;;) {
                    std::function<void()> task;
                    
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock,
                            [this]{ return this->stop || !this->tasks.empty(); });
                        if(this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    
                    task();
                }
            }
        );
}

// add new work item to the pool
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;
    
    auto task = std::make_shared< std::packaged_task<return_type()> >(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        
        // don't allow enqueueing after stopping the pool
        if(stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");
            
        tasks.emplace([task](){ (*task)(); });
    }
    condition.notify_one();
    return res;
}

// the destructor joins all threads
ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for(std::thread &worker: workers)
        worker.join();
}

void processAudioFile(const std::string& audioFile, std::vector<std::string>& results, std::mutex& results_mutex) {
    AlgorithmFactory& factory = standard::AlgorithmFactory::instance();
    std::unique_ptr<Algorithm> audio(factory.create("MonoLoader", "filename", audioFile));
    std::unique_ptr<Algorithm> pitchDetect(factory.create("PitchYin", "frameSize", 4096, "sampleRate", 44100, "tolerance", 0.8));
    std::vector<Real> audioBuffer;
    std::vector<Real> frame;
    Real pitch, confidence;

    audio->output("audio").set(audioBuffer);
    audio->compute();

    std::unique_ptr<Algorithm> fc(factory.create("FrameCutter", "frameSize", 4096, "hopSize", 512));
    fc->input("signal").set(audioBuffer);
    fc->output("frame").set(frame);

    pitchDetect->input("signal").set(frame);
    pitchDetect->output("pitch").set(pitch);
    pitchDetect->output("pitchConfidence").set(confidence);

    std::stringstream output;
    while (true) {
        fc->compute();
        if (frame.empty()) break;
        pitchDetect->compute();
        output << audioFile << "," << pitch << "," << confidence << "\n";
    }

    {
        std::lock_guard<std::mutex> lock(results_mutex);
        results.push_back(output.str());
    }
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    std::string wavFolder = "./wav/clips__train_valid/181920";
    if (!fs::is_directory(wavFolder)) {
        std::cerr << "Directory 'wav' not found." << std::endl;
        return 1;
    }

    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(wavFolder)) {
        if (entry.path().extension() == ".wav") {
            files.push_back(entry.path().string());
        }
    }

    essentia::init();
    ThreadPool pool(std::thread::hardware_concurrency());
    std::vector<std::future<void>> futures;
    std::vector<std::string> results;
    std::mutex results_mutex;

    for (auto& file : files) {
        futures.emplace_back(pool.enqueue(processAudioFile, file, std::ref(results), std::ref(results_mutex)));
    }

    for (auto& fut : futures) {
        fut.get();
    }

    essentia::shutdown();

    std::ofstream csvFile("all_pitches_15_trial2.csv");
    csvFile << "File,Pitch,Confidence\n";
    for (auto& result : results) {
        csvFile << result;
    }
    csvFile.close();

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "All files processed. Results saved to 'all_pitches.csv'." << std::endl;
    std::cout << "Total processing time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}
