
#include "CDDFuse.hpp"
#include <vector>
#include "infer/trt_infer.hpp"
#include "common/ilogger.hpp"

namespace CDDFUSE {

    using namespace std;


    class FuseImpl : public Fuser {
    public:
        ~FuseImpl() = default;

        bool startup(const std::string &engine_path, int gpuid){
            gpu_ = gpuid;
            TRT::set_device(gpuid);

            infer_model_ = TRT::load_infer(engine_path);
            if(infer_model_ == nullptr){
                INFOE("Load model failed: %s", engine_path.c_str());
                return false;
            }
            stream_ = infer_model_->get_stream();
            infer_model_->print();

            return true;
        }


        cv::Mat fuse(cv::Mat& img_rgb, cv::Mat& img_tir) override{
            
            cv::Mat rgb_med = img_rgb;
            cv::Mat tir_med = img_tir;

            // 绑定输入
            xin_rgb = infer_model_->input(0);
            xin_tir = infer_model_->input(1);
            xin_rgb->set_norm_mat(0, rgb_med);
            xin_tir->set_norm_mat(0, tir_med);

            // 绑定输出
            fusion_out = infer_model_->output(0);
 
            // 推理
            infer_model_->forward();

            
        }

    private:

        shared_ptr<TRT::Infer> infer_model_;
        shared_ptr<TRT::Tensor> xin_rgb;
        shared_ptr<TRT::Tensor> xin_tir;
        shared_ptr<TRT::Tensor> fusion_out;
        TRT::CUStream stream_ = nullptr;
        int gpu_ = 0;

    };

    shared_ptr<Fuser> create_Fuser(const std::string &engine_path,int gpuid){
        shared_ptr<FuseImpl> instance(new FuseImpl{});
        if(!instance->startup(engine_path, gpuid))
            instance.reset();
        return instance;
    }

}
