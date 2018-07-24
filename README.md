# VINS-Mono
![](https://images2015.cnblogs.com/blog/542140/201707/542140-20170708095920019-932150180.png)
     
     代码主要分为：
      前端(feature tracker),
      后端(sliding window, loop closure)，
      还加了初始化(visual-imu aligment)
      
![](https://images2018.cnblogs.com/blog/699318/201804/699318-20180414235214918-500793897.png)
      
      
# Feature tracker 特征跟踪
    这部分代码在feature_tracker包下面，主要是接收图像topic,
    使用KLT光流算法跟踪特征点，同时保持每一帧图像有最少的(100-300)个特征点。

    根据配置文件中的freq，确定每隔多久的时候，
    把检测到的特征点打包成/feature_tracker/featuretopic 发出去，

    要是没有达到发送的时间，这幅图像的feature就作为下一时刻的
    KLT追踪的特征点，就是不是每一副图像都要处理的，那样计算时间大了，
    而且数据感觉冗余，帧与帧之间图像的差距不会那么明显。

    这里的freq配置文件建议至少设置10，为了保证好的前端。
```c
void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ROS_DEBUG("processing camera %d", i);
        if (i != 1 || !STEREO_TRACK)
        //调用FeatureTracker的readImage
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)));
    }

    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
            //更新feature的ID
                completed |= trackerData[j].updateID(i);
        if (!completed)
            break;
    }
    
    //发布特征点topic
    if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time)) <= FREQ)
    {
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        //特征点的id，图像的(u,v)坐标
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point;
        sensor_msgs::ChannelFloat32 v_of_point;

        pub_img.publish(feature_points);

    }

     if (SHOW_TRACK)
     {
        //根据特征点被追踪的次数，显示他的颜色，越红表示这个特征点看到的越久，一幅图像要是大部分特征点是蓝色，前端tracker效果很差了，估计要挂了
        double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
        cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
     }


}

void FeatureTracker::readImage(const cv::Mat &_img)
{
    //直方图均匀化
    //if image is too dark or light, trun on equalize to find enough features
    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }

    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        //根据上一时刻的cur_img,cur_pts,寻找当前时刻的forw_pts,
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);
    }

    if (img_cnt == 0)
    {
        //根据fundamentalMatrix中的ransac去除一些outlier
        rejectWithF();
        //跟新特征点track的次数
        for (auto &n : track_cnt)
            n++;
        //为下面的goodFeaturesToTrack保证相邻的特征点之间要相隔30个像素,设置mask image
        setMask();

        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            //保证每个image有足够的特征点，不够就新提取
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.1, MIN_DIST, mask);
        }


    }
}

```
# 滑动窗口优化更新 Slide Window

    主要是：
       对imu的数据进行预积分，
       vision重投影误差的构造，
       loop-closure的检测，
       slide-window的维护 ，
       marginzation prior的维护，
       东西比较多。

    loop-closure的检测是使用视觉词带的，
    这里的特征不是feature-tracker的，那样子太少了。
    是通过订阅IMAGE_TOPIC,传递到闭环检测部分，重新检测的，
    这个我还没有认真看(做了很多限制，为了搜索的速度，词带不会很大，做了很多限制，
    从论文上看优化的方程只是加了几个vision重投影的限制，速度不会太慢)。

    是只有4个自由度的优化，roll, pitch由于重力对齐的原因是可观测的，就不去优化。

    最主要的还是下面这个最小二乘法方程构建，主要的代码我列出来。
![](https://images2015.cnblogs.com/blog/542140/201707/542140-20170708095941394-815386731.png)
    
```c
void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{

    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //调用imu的预积分，propagation ,计算对应的雅可比矩阵
        //if(solver_flag != NON_LINEAR)
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        //提供imu计算的当前位置，速度，作为优化的初值
        int j = frame_count;         
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }

}

void Estimator::processImage(const map<int, vector<pair<int, Vector3d>>> &image, const std_msgs::Header &header)
{
    //根据视差判断是不是关键帧，
    if (f_manager.addFeatureCheckParallax(frame_count, image))
        marginalization_flag = MARGIN_OLD;
    else
        marginalization_flag = MARGIN_SECOND_NEW;

    ImageFrame imageframe(image, header.stamp.toSec());
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(header.stamp.toSec(), imageframe));
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

//参数要是设置imu-camera的外参数未知，也可以帮你求解的
    if(ESTIMATE_EXTRINSIC == 2)
    {
    }

//初始化的流程
    if (solver_flag == INITIAL)
    {
        if (frame_count == WINDOW_SIZE)
        {
            bool result = false;
            if( ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1)
            {
            //构造sfm，优化imu偏差，加速度g，尺度的确定
               result = initialStructure();
               initial_timestamp = header.stamp.toSec();
            }
            if(result)
            {
                solver_flag = NON_LINEAR;
                solveOdometry();
                slideWindow();
                f_manager.removeFailures();
                ROS_INFO("Initialization finish!");
                last_R = Rs[WINDOW_SIZE];
                last_P = Ps[WINDOW_SIZE];
                last_R0 = Rs[0];
                last_P0 = Ps[0];
                
            }
            else
                slideWindow();
        }
    //先凑够window-size的数量的Frame
        else
            frame_count++;
    }
    else
    {
       
        solveOdometry();

//失败的检测
        if (failureDetection())
        {
            clearState();
            setParameter();
            return;
        }

        slideWindow();
        f_manager.removeFailures();
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }

}

void Estimator::slideWindow()
{
//WINDOW_SIZE中的参数的之间调整，同时FeatureManager进行管理feature，有些点要删除掉，有些点的深度要在下一frame表示(start frame已经删除了)


    Headers[frame_count - 1] = Headers[frame_count];
    Ps[frame_count - 1] = Ps[frame_count];
    Vs[frame_count - 1] = Vs[frame_count];
    Rs[frame_count - 1] = Rs[frame_count];
    Bas[frame_count - 1] = Bas[frame_count];
    Bgs[frame_count - 1] = Bgs[frame_count];

    delete pre_integrations[WINDOW_SIZE];
    pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};
//清楚数据，给下一副图像提供空间
    dt_buf[WINDOW_SIZE].clear();
    linear_acceleration_buf[WINDOW_SIZE].clear();
    angular_velocity_buf[WINDOW_SIZE].clear();    
}

void Estimator::solveOdometry()
{
    if (frame_count < WINDOW_SIZE)
        return;
    if (solver_flag == NON_LINEAR)
    {
        //三角化点
        f_manager.triangulate(Ps, tic, ric);
        ROS_DEBUG("triangulation costs %f", t_tri.toc());
        optimization();
    }
}

void Estimator::optimization()
{
    //添加frame的state，(p,v,q,b_a,b_g)，就是ceres要优化的参数
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    //添加camera-imu的外参数
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
    }

    //为ceres参数赋予初值
    vector2double();

    //添加margination residual， 先验知识
    //他的Evaluate函数看好，固定了线性化的点，First Jacobian Estimate
    if (last_marginalization_info)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    } 

    //添加imu的residual
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;
        IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }

    //添加vision的residual
    for (auto &it_per_id : f_manager.feature)
    {
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                continue;
            }
            Vector3d pts_j = it_per_frame.point;
            ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
            problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
            f_m_cnt++;
        }
    }

    //添加闭环的参数和residual
    if(LOOP_CLOSURE)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(front_pose.loop_pose, SIZE_POSE, local_parameterization);
        
        if(front_pose.features_ids[retrive_feature_index] == it_per_id.feature_id)
        {
            Vector3d pts_j = Vector3d(front_pose.measurements[retrive_feature_index].x, front_pose.measurements[retrive_feature_index].y, 1.0);
            Vector3d pts_i = it_per_id.feature_per_frame[0].point;
            
            ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
            problem.AddResidualBlock(f, loss_function, para_Pose[start], front_pose.loop_pose, para_Ex_Pose[0], para_Feature[feature_index]);
        
            retrive_feature_index++;
            loop_factor_cnt++;
        }
    }


    //设置了优化的最长时间，保证实时性
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;

    // 求解
    ceres::Solve(options, &problem, &summary);

// http://blog.csdn.net/heyijia0327/article/details/53707261#comments
// http://blog.csdn.net/heyijia0327/article/details/52822104
    if (marginalization_flag == MARGIN_OLD)
    {
        //如果当前帧是关键帧的，把oldest的frame所有的信息margination，作为下一时刻的先验知识，参考上面的两个网址，大神的解释很明白

    }
    else{
        //如果当前帧不是关键帧的，把second newest的frame所有的视觉信息丢弃掉，imu信息不丢弃，记住不是做margination，是为了保持矩阵的稀疏性
    }
    
}

```
# 后续

     imu的参数很重要，还有就是硬件同步，global shutter的摄像头很重要。
     我要是动作快的话，效果就不行了。但人家的视频感觉效果很不错。

     这个还要继续弄硬件和代码原理，
     代码中最小二乘法优化中的FOCAL_LENGTH感觉要根据自己的摄像头设置，
     还没有具体看，视觉信息矩阵的设置还没有看。

     工程中具体的情况还是要自己解决，

## A Robust and Versatile Monocular Visual-Inertial State Estimator
**29 Dec 2017**: New features: Add map merge, pose graph reuse, online temporal calibration function, and support rolling shutter camera. Map reuse videos: 

<a href="https://www.youtube.com/embed/WDpH80nfZes" target="_blank"><img src="http://img.youtube.com/vi/WDpH80nfZes/0.jpg" 
alt="cla" width="240" height="180" border="10" /></a>
<a href="https://www.youtube.com/embed/eINyJHB34uU" target="_blank"><img src="http://img.youtube.com/vi/eINyJHB34uU/0.jpg" 
alt="icra" width="240" height="180" border="10" /></a>

VINS-Mono is a real-time SLAM framework for **Monocular Visual-Inertial Systems**. It uses an optimization-based sliding window formulation for providing high-accuracy visual-inertial odometry. It features efficient IMU pre-integration with bias correction, automatic estimator initialization, online extrinsic calibration, failure detection and recovery, loop detection, and global pose graph optimization, map merge, pose graph reuse, online temporal calibration, rolling shutter support. VINS-Mono is primarily designed for state estimation and feedback control of autonomous drones, but it is also capable of providing accurate localization for AR applications. This code runs on **Linux**, and is fully integrated with **ROS**. For **iOS** mobile implementation, please go to [VINS-Mobile](https://github.com/HKUST-Aerial-Robotics/VINS-Mobile).

**Authors:** [Tong Qin](http://www.qintonguav.com), [Peiliang Li](https://github.com/PeiliangLi), [Zhenfei Yang](https://github.com/dvorak0), and [Shaojie Shen](http://www.ece.ust.hk/ece.php/profile/facultydetail/eeshaojie) from the [HUKST Aerial Robotics Group](http://uav.ust.hk/)

**Videos:**

<a href="https://www.youtube.com/embed/mv_9snb_bKs" target="_blank"><img src="http://img.youtube.com/vi/mv_9snb_bKs/0.jpg" 
alt="euroc" width="240" height="180" border="10" /></a>
<a href="https://www.youtube.com/embed/g_wN0Nt0VAU" target="_blank"><img src="http://img.youtube.com/vi/g_wN0Nt0VAU/0.jpg" 
alt="indoor_outdoor" width="240" height="180" border="10" /></a>
<a href="https://www.youtube.com/embed/I4txdvGhT6I" target="_blank"><img src="http://img.youtube.com/vi/I4txdvGhT6I/0.jpg" 
alt="AR_demo" width="240" height="180" border="10" /></a>

EuRoC dataset;                  Indoor and outdoor performance;                         AR application;

<a href="https://www.youtube.com/embed/2zE84HqT0es" target="_blank"><img src="http://img.youtube.com/vi/2zE84HqT0es/0.jpg" 
alt="MAV platform" width="240" height="180" border="10" /></a>
<a href="https://www.youtube.com/embed/CI01qbPWlYY" target="_blank"><img src="http://img.youtube.com/vi/CI01qbPWlYY/0.jpg" 
alt="Mobile platform" width="240" height="180" border="10" /></a>

 MAV application;               Mobile implementation (Video link for mainland China friends: [Video1](http://www.bilibili.com/video/av10813254/) [Video2](http://www.bilibili.com/video/av10813205/) [Video3](http://www.bilibili.com/video/av10813089/) [Video4](http://www.bilibili.com/video/av10813325/) [Video5](http://www.bilibili.com/video/av10813030/))

**Related Papers**
* **VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator**, Tong Qin, Peiliang Li, Zhenfei Yang, Shaojie Shen [arXiv:1708.03852](https://arxiv.org/abs/1708.03852v1) 
* **Autonomous Aerial Navigation Using Monocular Visual-Inertial Fusion**, Yi Lin, Fei Gao, Tong Qin, Wenliang Gao, Tianbo Liu, William Wu, Zhenfei Yang, Shaojie Shen, J Field Robotics. 2017;00:1–29. [https://doi.org/10.1002/rob.21732](https://doi.org/10.1002/rob.21732)  
```
@article{qin2017vins,
  title={VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator},
  author={Qin, Tong and Li, Peiliang and Shen, Shaojie},
  journal={arXiv preprint arXiv:1708.03852},
  year={2017}
}
```
```
@article{Lin17,
  Author = {Y. Lin and F. Gao and T. Qin and W. Gao and T. Liu and W. Wu and Z. Yang and S. Shen},
  Journal = jfr,
  Title = {Autonomous Aerial Navigation Using Monocular Visual-Inertial Fusion},  
  Volume = {00},
  Pages = {1-29},
  Year = {2017}} 
```
*If you use VINS-Mono for your academic research, please cite at least one of our related papers.*

## 1. Prerequisites
1.1 **Ubuntu** and **ROS**
Ubuntu  16.04.
ROS Kinetic. [ROS Installation](http://wiki.ros.org/ROS/Installation)
additional ROS pacakge
```
    sudo apt-get install ros-YOUR_DISTRO-cv-bridge ros-YOUR_DISTRO-tf ros-YOUR_DISTRO-message-filters ros-YOUR_DISTRO-image-transport
```


1.2. **Ceres Solver**
Follow [Ceres Installation](http://ceres-solver.org/installation.html), remember to **make install**.
(Our testing environment: Ubuntu 16.04, ROS Kinetic, OpenCV 3.3.1, Eigen 3.3.3) 

## 2. Build VINS-Mono on ROS
Clone the repository and catkin_make:
```
    cd ~/catkin_ws/src
    git clone https://github.com/HKUST-Aerial-Robotics/VINS-Mono.git
    cd ../
    catkin_make
    source ~/catkin_ws/devel/setup.bash
```

## 3. Visual-Inertial Odometry and Pose Graph Reuse on Public datasets
Download [EuRoC MAV Dataset](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets). Although it contains stereo cameras, we only use one camera. The system also works with [ETH-asl cla dataset](http://robotics.ethz.ch/~asl-datasets/maplab/multi_session_mapping_CLA/bags/). We take EuRoC as the example.

**3.1 visual-inertial odometry and loop closure**

3.1.1 Open three terminals, launch the vins_estimator , rviz and play the bag file respectively. Take MH_01 for example
```
    roslaunch vins_estimator euroc.launch 
    roslaunch vins_estimator vins_rviz.launch
    rosbag play YOUR_PATH_TO_DATASET/MH_01_easy.bag 
```
(If you fail to open vins_rviz.launch, just open an empty rviz, then load the config file: file -> Open Config-> YOUR_VINS_FOLDER/config/vins_rviz_config.rviz)

3.1.2 (Optional) Visualize ground truth. We write a naive benchmark publisher to help you visualize the ground truth. It uses a naive strategy to align VINS with ground truth. Just for visualization. not for quantitative comparison on academic publications.
```
    roslaunch benchmark_publisher publish.launch  sequence_name:=MH_05_difficult
```
 (Green line is VINS result, red line is ground truth). 
 
3.1.3 (Optional) You can even run EuRoC **without extrinsic parameters** between camera and IMU. We will calibrate them online. Replace the first command with:
```
    roslaunch vins_estimator euroc_no_extrinsic_param.launch
```
**No extrinsic parameters** in that config file.  Waiting a few seconds for initial calibration. Sometimes you cannot feel any difference as the calibration is done quickly.

**3.2 map merge**

After playing MH_01 bag, you can continue playing MH_02 bag, MH_03 bag ... The system will merge them according to the loop closure.

**3.3 map reuse**

3.3.1 map save

Set the **pose_graph_save_path** in the config file (YOUR_VINS_FOLEDER/config/euroc/euroc_config.yaml). After playing MH_01 bag, input **s** in vins_estimator terminal, then **enter**. The current pose graph will be saved. 

3.3.2 map load

Set the **load_previous_pose_graph** to 1 before doing 3.1.1. The system will load previous pose graph from **pose_graph_save_path**. Then you can play MH_02 bag. New sequence will be aligned to the previous pose graph.

## 4. AR Demo
4.1 Download the [bag file](https://www.dropbox.com/s/s29oygyhwmllw9k/ar_box.bag?dl=0), which is collected from HKUST Robotic Institute. For friends in mainland China, download from [bag file](https://pan.baidu.com/s/1geEyHNl).

4.2 Open three terminals, launch the ar_demo, rviz and play the bag file respectively.
```
    roslaunch ar_demo 3dm_bag.launch
    roslaunch ar_demo ar_rviz.launch
    rosbag play YOUR_PATH_TO_DATASET/ar_box.bag 
```
We put one 0.8m x 0.8m x 0.8m virtual box in front of your view. 

## 5. Run with your device 

Suppose you are familiar with ROS and you can get a camera and an IMU with raw metric measurements in ROS topic, you can follow these steps to set up your device. For beginners, we highly recommend you to first try out [VINS-Mobile](https://github.com/HKUST-Aerial-Robotics/VINS-Mobile) if you have iOS devices since you don't need to set up anything.

5.1 Change to your topic name in the config file. The image should exceed 20Hz and IMU should exceed 100Hz. Both image and IMU should have the accurate time stamp. IMU should contain absolute acceleration values including gravity.

5.2 Camera calibration:

We support the [pinhole model](http://docs.opencv.org/2.4.8/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html) and the [MEI model](http://www.robots.ox.ac.uk/~cmei/articles/single_viewpoint_calib_mei_07.pdf). You can calibrate your camera with any tools you like. Just write the parameters in the config file in the right format. If you use rolling shutter camera, please carefully calibrate your camera, making sure the reprojection error is less than 0.5 pixel.

5.3 **Camera-Imu extrinsic parameters**:

If you have seen the config files for EuRoC and AR demos, you can find that we can estimate and refine them online. If you familiar with transformation, you can figure out the rotation and position by your eyes or via hand measurements. Then write these values into config as the initial guess. Our estimator will refine extrinsic parameters online. If you don't know anything about the camera-IMU transformation, just ignore the extrinsic parameters and set the **estimate_extrinsic** to **2**, and rotate your device set at the beginning for a few seconds. When the system works successfully, we will save the calibration result. you can use these result as initial values for next time. An example of how to set the extrinsic parameters is in[extrinsic_parameter_example](https://github.com/HKUST-Aerial-Robotics/VINS-Mono/blob/master/config/extrinsic_parameter_example.pdf)

5.4 **Temporal calibration**:
Most self-made visual-inertial sensor sets are unsynchronized. You can set **estimate_td** to 1 to online estimate the time offset between your camera and IMU.  

5.5 **Rolling shutter**:
For rolling shutter camera (carefully calibrated, reprojection error under 0.5 pixel), set **rolling_shutter** to 1. Also, you should set rolling shutter readout time **rolling_shutter_tr**, which is from sensor datasheet(usually 0-0.05s, not exposure time). Don't try web camera, the web camera is so awful.

5.6 Other parameter settings: Details are included in the config file.

5.7 Performance on different devices: 

(global shutter camera + synchronized high-end IMU, e.g. VI-Sensor) > (global shutter camera + synchronized low-end IMU) > (global camera + unsync high frequency IMU) > (global camera + unsync low frequency IMU) > (rolling camera + unsync low frequency IMU). 


## 6. Acknowledgements
We use [ceres solver](http://ceres-solver.org/) for non-linear optimization and [DBoW2](https://github.com/dorian3d/DBoW2) for loop detection, and a generic [camera model](https://github.com/hengli/camodocal).

## 7. Licence
The source code is released under [GPLv3](http://www.gnu.org/licenses/) license.

We are still working on improving the code reliability. For any technical issues, please contact Tong QIN <tong.qinATconnect.ust.hk> or Peiliang LI <pliapATconnect.ust.hk>.

For commercial inquiries, please contact Shaojie SHEN <eeshaojieATust.hk>
