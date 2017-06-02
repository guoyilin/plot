#include <stdio.h>
#include <string>
#include <vector>
#include <map>
#include <math.h>
#include <iostream>
#include <fstream>
#include "caffe/caffe.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


struct ImgRect {
    int x1; 
    int y1;
    int x2;
    int y2;
    float score;
    ImgRect(int _x1, int _y1, int _x2, int _y2, float _score) : x1(_x1), y1(_y1), x2(_x2) , y2(_y2), score(_score) {}
};
//贪心法nms, 假定proposals是降序, 只返回topN
void nms(vector<int>& keep, float* proposals, int proposals_len,
                                    float nms_thresh) {
    vector<int> order(proposals_len, 0);
    for (int i = 0; i < order.size(); i++)
        order[i] = i;
    //compute all areas for all proposals.
    vector<float> areas;
    for (int i = 0; i < proposals_len; i++) {
        areas.push_back(
                        (proposals[i * 5 + 2] - proposals[i * 5 + 0] + 1)
                        * (proposals[i * 5 + 3] - proposals[i * 5 + 1] + 1));
    }
    
    while (order.size() > 0) {
        int i = order[0];
        // float area = (proposals[i*5 + 2] - proposals[i*5 + 0] + 1) * (proposals[i*5 + 3] - proposals[i*5 + 1] + 1);
        keep.push_back(i); //保留
        //计算 i proposal   和其他order中剩余的所有proposals的iou.
        vector<int> next_order;
        for (int j = 1; j < order.size(); j++) {
            float xx1 = max(proposals[i * 5 + 0], proposals[order[j] * 5 + 0]);
            float yy1 = max(proposals[i * 5 + 1], proposals[order[j] * 5 + 1]);
            float xx2 = min(proposals[i * 5 + 2], proposals[order[j] * 5 + 2]);
            float yy2 = min(proposals[i * 5 + 3], proposals[order[j] * 5 + 3]);
            float w = max(0.0, xx2 - xx1 + 1);
            float h = max(0.0, yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (areas[i] + areas[order[j]] - inter);
            if (ovr < nms_thresh) {
                next_order.push_back(order[j]);
            }
            
        }
        order = next_order; //copy
    }
    
}
/*
 * ===  FUNCTION  ======================================================================
 *         Name:  bbox_transform_inv
 *  Description:  Compute bounding box regression value
 * =====================================================================================
 */
void bbox_transform_inv(int num, const float* box_deltas,
                                                   const float* pred_cls, float* boxes, float* pred, int img_height,
                                                   int img_width) {
    float width, height, ctr_x, ctr_y, dx, dy, dw, dh, pred_ctr_x, pred_ctr_y,
    pred_w, pred_h;
    for (int i = 0; i < num; i++) //for each box.
    {
        width = boxes[i * 4 + 2] - boxes[i * 4 + 0] + 1.0; //box width
        height = boxes[i * 4 + 3] - boxes[i * 4 + 1] + 1.0; //box height
        ctr_x = boxes[i * 4 + 0] + 0.5 * width; //center of box.
        ctr_y = boxes[i * 4 + 1] + 0.5 * height; //center of box.
        for (int j = 0; j < class_num; j++) //for each class. we have two class.
        {
            
            dx = box_deltas[(i * class_num + j) * 4 + 0];
            dy = box_deltas[(i * class_num + j) * 4 + 1];
            dw = box_deltas[(i * class_num + j) * 4 + 2];
            dh = box_deltas[(i * class_num + j) * 4 + 3];
            pred_ctr_x = ctr_x + width * dx;
            pred_ctr_y = ctr_y + height * dy;
            pred_w = width * exp(dw);
            pred_h = height * exp(dh);
            pred[(j * num + i) * 5 + 0] = max(
                                              min(pred_ctr_x - 0.5* pred_w, img_width -1), 0);
            pred[(j * num + i) * 5 + 1] = max(
                                              min(pred_ctr_y - 0.5* pred_h, img_height -1), 0);
            pred[(j * num + i) * 5 + 2] = max(
                                              min(pred_ctr_x + 0.5* pred_w, img_width -1), 0);
            pred[(j * num + i) * 5 + 3] = max(
                                              min(pred_ctr_y + 0.5* pred_h, img_height -1), 0);
            pred[(j * num + i) * 5 + 4] = pred_cls[i * class_num + j];
        }
    }
}


/*
 * ===  FUNCTION  ======================================================================
 *         Name:  boxes_sort
 *  Description:  Sort the bounding box according score
 * =====================================================================================
 */
void boxes_sort(const int num, const float* pred,
                                           float* sorted_pred) {
    vector<Info> my;
    Info tmp;
    for (int i = 0; i < num; i++) {
        tmp.score = pred[i * 5 + 4];
        tmp.head = pred + i * 5;
        my.push_back(tmp);
    }
    std::sort(my.begin(), my.end(), compareScore);
    for (int i = 0; i < num; i++) {
        for (int j = 0; j < 5; j++)
            sorted_pred[i * 5 + j] = my[i].head[j];
    }
}
/*
 * ===  FUNCTION  ======================================================================
 *         Name:  Detection_Pvanet
 *        Input:  img [ confidence nms_thres max_input_size min_input_size]
 *  Description:  Detect the object and return the result (type: string), only for pvanet.
 *                The img will be resized by 32 x N
 * =====================================================================================
 */
vector<ImgRect> Detection_Pvanet(cv::Mat &cv_img,  float CONF_THRESH, float NMS_THRESH, const int max_input_side, const int min_input_side) {
    vector<ImgRect> rects;
    if (cv_img.empty()) {
        LOG(ERROR) << "Can not get the image file: " << img_name;
        return rects;
    }
    cv::Mat cv_new(cv_img.rows, cv_img.cols, CV_32FC3, cv::Scalar(0, 0, 0));
    //resize img
    int max_side = max(cv_img.rows, cv_img.cols);
    int min_side = min(cv_img.rows, cv_img.cols);
    float img_scale = float(min_input_side) / float(min_side);
    //Prevent the biggest axis from being more than MAX_SIZE
    float im_scale_x = 0;
    float im_scale_y = 0;
    if (img_scale * max_side > max_input_side)
        img_scale = float(max_input_side) / float(max_side);
    
    im_scale_x = floor(cv_img.cols * img_scale / 32) * 32 / cv_img.cols;
    im_scale_y = floor(cv_img.rows * img_scale / 32) * 32 / cv_img.rows;
    const int height = int(cv_img.rows * im_scale_y);
    const int width = int(cv_img.cols * im_scale_x);
    
    int num_out;
    
    cv::Mat cv_resized;
    //std::cout << "imagename " << im_name << endl;
    float im_info[3];
    float *data_buf = new float[height * width * 3]; //输入数据
    float *boxes = NULL;
    float *pred = NULL;
    float *pred_per_class = NULL;
    float *sorted_pred_cls = NULL;
    const float* bbox_delt;
    const float* rois;
    const float* pred_cls;
    int num;
    //原始图片每个像素减去mean
    for (int h = 0; h < cv_img.rows; ++h) {
        for (int w = 0; w < cv_img.cols; ++w) {
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[0] = float(
                                                             cv_img.at<cv::Vec3b>(cv::Point(w, h))[0]) - float(102.9801);
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[1] = float(
                                                             cv_img.at<cv::Vec3b>(cv::Point(w, h))[1]) - float(115.9465);
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[2] = float(
                                                             cv_img.at<cv::Vec3b>(cv::Point(w, h))[2]) - float(122.7717);
        }
    }
    if(width ==0 || height == 0)
    {
        LOG(ERROR) << "height or width= 0: "  << img_name;
        return rects;
    }
    cv::resize(cv_new, cv_resized, cv::Size(width, height));
    
    im_info[0] = cv_resized.rows;
    im_info[1] = cv_resized.cols;
    im_info[2] = img_scale;
    //std::cout << "im_info: " << im_info[2] << std::endl;
    //准备数据
    //    imwrite("resize.jpg", cv_resized);
    //std::cout << im_info[0] << " " << im_info[1] << " " << im_info[2] << std::endl;
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            data_buf[(0 * height + h) * width + w] = float(
                                                           cv_resized.at<cv::Vec3f>(cv::Point(w, h))[0]);
            data_buf[(1 * height + h) * width + w] = float(
                                                           cv_resized.at<cv::Vec3f>(cv::Point(w, h))[1]);
            data_buf[(2 * height + h) * width + w] = float(
                                                           cv_resized.at<cv::Vec3f>(cv::Point(w, h))[2]);
        }
    }
    //开始forward
    net_->blob_by_name("data")->Reshape(1, 3, height, width);
    net_->Reshape();
    boost::shared_ptr<Blob<float> > data_layer = net_->blob_by_name("data");
    float* input_data = data_layer->mutable_cpu_data();
    memcpy ( input_data, data_buf, height*width*3*sizeof(float));
    boost::shared_ptr<Blob<float> > im_layer = net_->blob_by_name("im_info");
    float *input_im = im_layer->mutable_cpu_data();
    input_im[0] = im_info[0];
    input_im[1] = im_info[1];
    input_im[2] = im_info[2];
    
    double t_start = cv::getTickCount();
    net_->ForwardFrom(0);
    LOG(INFO) << "附加方牌检测， Forward time:" << (cv::getTickCount() - t_start)/cv::getTickFrequency()*1000;
    //得到回归结果和分类结果
    bbox_delt = net_->blob_by_name("bbox_pred")->cpu_data();
    num = net_->blob_by_name("rois")->num();
    rois = net_->blob_by_name("rois")->cpu_data();
    pred_cls = net_->blob_by_name("cls_prob")->cpu_data();
    //根据分类score，排序得到>threshold的box
    boxes = new float[num * 4];
    //最终的坐标位置和分类score
    pred = new float[num * 5 * class_num];
    pred_per_class = new float[num * 5];//存储当前类的预测位置和预测值
    sorted_pred_cls = new float[num * 5];
    
    //box坐标变换为原始图片坐标
    for (int n = 0; n < num; n++) {
        for (int c = 0; c < 4; c++) {
            //[n*5+c] is score???
            if( c == 0 || c == 2)
                boxes[n * 4 + c] = rois[n * 5 + c + 1] / im_scale_x;
            else
                boxes[n * 4 + c] = rois[n * 5 + c + 1] / im_scale_y;
        }
    }
    //box坐标位移变化
    bbox_transform_inv(num, bbox_delt, pred_cls, boxes, pred, cv_img.rows,
                       cv_img.cols);
    
    int total_detect_num = 0;
    for (int i = 1; i < class_num; i++) {
        for (int j = 0; j < num; j++) {
            for (int k = 0; k < 5; k++){
                pred_per_class[j * 5 + k] = pred[(i * num + j) * 5 + k];
            }
        }
        boxes_sort(num, pred_per_class, sorted_pred_cls);
        vector<int> keep;
        nms(keep, sorted_pred_cls, num,  NMS_THRESH);
        string classname = CLASSES[i];
        
        //vis_detections(cv_img, keep, sorted_pred_cls, CONF_THRESH, classname, fout, im_name);//保存当前类的检测结果
        int ii = 0;
        //  while(i < keep.size()) {
        // std::cout << "keep:" << keep.size() << std::endl;
        while (ii < keep.size()) {
            if(sorted_pred_cls[keep[ii] * 5 + 4] < CONF_THRESH){
                ii++;
                continue;
            }
            if (ii >= keep.size() - 1)
                break;
#if SHOW_IMG
            cv::rectangle(cv_img,
                          cv::Point(sorted_pred_cls[keep[ii] * 5 + 0],
                                    sorted_pred_cls[keep[ii] * 5 + 1]),
                          cv::Point(sorted_pred_cls[keep[ii] * 5 + 2],
                                    sorted_pred_cls[keep[ii] * 5 + 3]),
                          cv::Scalar(0, 0, 255));
#endif
            total_detect_num++;
            ImgRect result_rect(static_cast<int>(sorted_pred_cls[keep[ii] * 5 + 0]),
                                static_cast<int>(sorted_pred_cls[keep[ii] * 5 + 1]) ,
                                static_cast<int>(sorted_pred_cls[keep[ii] * 5 + 2]),
                                static_cast<int>(sorted_pred_cls[keep[ii] * 5 + 3]),
                                static_cast<float>(sorted_pred_cls[keep[ii] * 5 + 4]));
            rects.push_back(result_rect);
            ii++;
        }
    }
    //std::cout << "total_detect_num:"  << total_detect_num << std::endl;
    //std::cout << rects.size() << std::endl;
#if SHOW_IMG
    cv::imwrite("test_result/" + img_name, cv_img);
    
    //    cv::imshow("img", cv_img);
    //    cv::waitKey(0);
#endif
    //cv::imwrite(saveResult, cv_img);
    //std::cout << "finished:" << saveResult << std::endl;
    delete[] data_buf;
    delete[] boxes;
    delete[] pred;
    delete[] pred_per_class;
    delete[] sorted_pred_cls;
    return rects;
}


bool fjfp_detection(const string& im_name, vector<ImgRect>& rects)
{
    //fasterrcnn检测参数
    float LOW_THRESH = 0.4;
    float HIGH_THRESH = 0.6;
    float NMS_THRESH = 0.3;
    int max_input_side = 1000;
    int min_input_side = 600;
    cv::Mat cv_img = cv::imread(im_name);
    
    rects = Detection_Pvanet(cv_img,  LOW_THRESH,NMS_THRESH, max_input_side, min_input_side);
    
    return true;
}
