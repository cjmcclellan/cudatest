//
// Created by connor on 7/30/21.
//


#include "loadmodels_incuda.h"

std::string GPUDeviceName(tensorflow::Session* session) {
    std::vector<tensorflow::DeviceAttributes> devices;
    TF_CHECK_OK(session->ListDevices(&devices));
    for (const tensorflow::DeviceAttributes& d : devices) {
        if (d.device_type() == "GPU" || d.device_type() == "gpu") {
            return d.name();
        }
    }
    return "";
}

void printCUDADS(double *ptr, std::string name, std::size_t size, int num){
    double test[num];
    cudaMemcpy(&test, ptr, size * num, cudaMemcpyDeviceToHost);
    std::cout << name << " value is :";
    for (int i = 0; i < num; i++) {
        std::cout << " " << test[i] << ", ";
    }
    std::cout << "\n";
}

void runModel(void) {

//    std::string PathGraph = "/home/connor/Documents/DeepSim/CUDA/TFCPP/models/resistor";
//    std::string PathGraph = "/home/connor/Documents/DeepSim/SPICE/cuspice/models/matrix6dummy/tfmodel";
//    std::string PathGraph = "/home/connor/Documents/DeepSim/SPICE/cuspice/models/matrix6con/tfmodel";
//    std::string PathGraph = "/home/connor/Documents/DeepSim/SPICE/cuspice/models/matrix6conbatch/tfmodel";
//    std::string PathGraph = "/home/deepsim/Documents/SPICE/DSSpice/src/deepsim/models/matrix6conandt/tfmodel";
    std::string PathGraph = "/home/connor/Documents/DeepSim/AI/thermal-nn-tests/data/OpenRoadDesigns/asap7/asapmodels/54nm/models/symmetric/and2x2_asap7_75t_r/tfmodel";
    const int num = 2;
    int numNodes = 23;
    int numOutputs = 23;

    std::string inputLayer = "serving_default_input_temperature:0";
    std::string outputLayer = "PartitionedCall:1";

    // create a session that takes our
    // scope as the root scope
    tensorflow::Status status;
//    tensorflow::GraphDef graph_def;
    tensorflow::SessionOptions session_options;
    tensorflow::RunOptions run_options;
    tensorflow::SavedModelBundleLite* bundle = new tensorflow::SavedModelBundleLite();
    session_options.config.mutable_gpu_options()->set_allow_growth(true);
    tensorflow::Status load_graph_status = tensorflow::LoadSavedModel(session_options,
                                                                      run_options,
                                                                      PathGraph,
                                                                      {"serve"},
                                                                      bundle);
    tensorflow::Session * session = bundle->GetSession();
    std::vector<tensorflow::Tensor> outputs;
    const std::string gpu_device_name = GPUDeviceName(session);
    // add the input layer to the session
    tensorflow::CallableOptions opts;
    tensorflow::Session::CallableHandle feed_gpu_fetch_cpu;
    opts.add_feed(inputLayer);
    opts.set_fetch_skip_sync(true);
    opts.add_fetch(outputLayer);
    opts.clear_fetch_devices();
    opts.mutable_feed_devices()->insert({inputLayer, gpu_device_name});
    opts.mutable_fetch_devices()->insert({outputLayer, gpu_device_name});
    session->MakeCallable(opts, &feed_gpu_fetch_cpu);

    tensorflow::PlatformDeviceId gpu_id(0);
    auto *allocator = new tensorflow::GPUcudaMallocAllocator(gpu_id);
    auto input_tensor = tensorflow::Tensor(allocator, tensorflow::DT_DOUBLE,
                                           tensorflow::TensorShape({num, numNodes}));
    // load the model
//    SavedModelBundleLite bundle;
//    SessionOptions session_options;
//    RunOptions run_options;
//    session_options.config.mutable_gpu_options()->set_allow_growth(true);
//    std::cout << "DebugString -> " << status.error_message() << std::endl;

//    std::vector<Tensor> input_data[num] = {};
    typedef double T;
//    Tensor input_data(tensorflow::DT_DOUBLE,tensorflow::TensorShape({num, numNodes}));
    std::vector<double> h_input(num * numNodes);
    std::vector<double> h_output(num * numOutputs);

    double val = 100.0;
    double val2 = 1.0;
    for (int i = 0; i < num; i++){
        for (int j = 0; j < numNodes; j++) {
            if (j < numNodes / 2 && i < 2)
                h_input[i * numNodes + j] = val2;
            else
                h_input[i * numNodes + j] = val;
        }

    }

    cudaMemcpy(input_tensor.flat<double>().data(), &h_input[0], num * numNodes * sizeof(double), cudaMemcpyHostToDevice);
//
    printCUDADS(input_tensor.flat<double>().data(), "\ninput before", sizeof(double), num * numNodes);

    std::cout << "\ninputs\n";
    for (int i = 0; i < numNodes * num; i++){
        auto a = h_input[i];
        if (i % numNodes == 0){
            std::cout << "\n";
        }
        std::cout << a << " ";
//        a = predicted_boxes(1, i);
//        std::cout << a << " ";
    }
//    std::cout << "\n";
//    const string input_node = "serving_default_dense_input:0";


    status = session->RunCallable(feed_gpu_fetch_cpu,
                                  {input_tensor},
                                  &(outputs),
                                  nullptr);
    if (!status.ok())
    {
        LOG(ERROR) << "Running model failed: " << status;
    }

    double* output_tensor_data = outputs[0].flat<double>().data();
    printCUDADS(output_tensor_data, "output before", sizeof(double), num * numOutputs);

    cudaMemcpy(&h_output[0], output_tensor_data, num * numOutputs * sizeof(double), cudaMemcpyDeviceToHost);

//    auto predicted_scores = predictions[1].flat<double>();
//    auto predicted_labels = predictions[2].tensor<double, 2>();
//    status = ReadBinaryProto(tensorflow::Env::Default(),PathGraph, &graph_def);

//    status = session.Create(graph_def);
    std::cout << " \n outputs \n";
    for (int i = 0; i < numOutputs * num; i++){
        auto a = h_output[i];
        if (i % numOutputs == 0){
            std::cout << "\n";
        }
        std::cout << a << " ";
    }
}
