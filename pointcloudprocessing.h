#ifndef POINTCLOUDPROCESSING_H
#define POINTCLOUDPROCESSING_H
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/time.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <boost/thread/thread.hpp>
#include <pcl/point_types.h>
#include <pcl/common/common_headers.h>
#include <pcl/correspondence.h>
#include <pcl/common/transforms.h>
#include <pcl/common/transformation_from_correspondences.h>
#include <pcl/common/eigen.h>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef float f32;
typedef int s16;

class PointCloudProcessing
{
public:
    PointCloudProcessing();
    void getKnnRadius(const PointCloudT::Ptr &cloud,
                      const PointCloudT::Ptr &ptQuery,
                      const f32 searchRadius,
                      std::vector< std::vector<s16> > &neighIdx,
                      std::vector< std::vector<f32> > &neighDist);

    void pclRegionGrow(PointCloudT::Ptr scene,
                       PointCloudT::Ptr seeds,
                       float growSpeed,
                       float searchRadius,
                       float heightThd,
                       PointCloudT::Ptr &cloud_out,
                       std::set<int> &clusterIdx);

    void register2ScenesMEstimator( const PointCloudT::Ptr sceneRef,
                                    const PointCloudT::Ptr sceneNew,
                                    const PointCloudT::Ptr corrRef,
                                    const PointCloudT::Ptr corrNew,
                                    const float inlrThd,
                                    const float smpRate,
                                    const float inlrRate,
                                    const int maxIter,
                                    Eigen::Matrix4f &transMat,
                                    PointCloudT::Ptr cloudOut);
    void register2ScenesRansac( const PointCloudT::Ptr sceneRef,
                                const PointCloudT::Ptr sceneNew,
                                const PointCloudT::Ptr corrRef,
                                const PointCloudT::Ptr corrNew,
                                const float inlrThd,
                                const int sampleNb,
                                const float inlrRate,
                                const int maxIter,
                                Eigen::Matrix4f &transMat,
                                PointCloudT::Ptr cloudOut);

    void getTransformation3d( const PointCloudT::Ptr corrRef,
                              const PointCloudT::Ptr corrNew,
                              const std::vector<int> inlrIdx,
                              Eigen::Matrix4f &transMat);

    void getTransformation3d( const PointCloudT::Ptr corrRef,
                              const PointCloudT::Ptr corrNew,
                              const std::set<int> sampleIdx,
                              const float inlrThd,
                              Eigen::Matrix4f &transMat,
                              std::vector<float>& transDist,
                              std::vector<int>& inlrIdx,
                              int &inlrNb);
    void getTransform3dGeometric(const PointCloudT::Ptr corrRef,
                                 const PointCloudT::Ptr corrNew,
                                 const std::set<int> sampleIdx,
                                 const float inlrThd,
                                 Eigen::Matrix4f &transMat,
                                 std::vector<float>& transDist,
                                 std::vector<int>& inlrIdx,
                                 int &inlrNb);

    void normalizePointClouds(PointCloudT::Ptr &corrRef,
                              Eigen::Matrix4f &transMat);
    void getRTGeometricLinearSystem(PointCloudT::Ptr corrRef,
                                    PointCloudT::Ptr corrNew,
                                    Eigen::Matrix4f &transMat);
    void randIdx(int idxRange, int sampleNb, std::time_t t, std::set<int>& randSmp);
    void getMaxIdx(const std::vector<int> vec, int &maxIdx);
};

#endif // POINTCLOUDPROCESSING_H
