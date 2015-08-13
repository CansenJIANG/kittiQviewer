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
typedef pcl::PointXYZRGB PointC;
typedef pcl::PointCloud<PointC> PointCloudC;
typedef float f32;
typedef int s16;

class PointCloudProcessing
{
public:
    PointCloudProcessing();
    void getKnnRadius(const PointCloudC::Ptr &cloud,
                      const PointCloudC::Ptr &ptQuery,
                      const f32 searchRadius,
                      std::vector< std::vector<s16> > &neighIdx,
                      std::vector< std::vector<f32> > &neighDist);
    void removeBadSeed(PointCloudC::Ptr &cloud,
                        const f32 searchRadius);

    void pclRegionGrow(PointCloudC::Ptr scene,
                       PointCloudC::Ptr seeds,
                       float growSpeed,
                       float searchRadius,
                       float heightThd,
                       PointCloudC::Ptr &cloud_out,
                       std::set<int> &clusterIdx);

    void register2ScenesMEstimator( const PointCloudC::Ptr sceneRef,
                                    const PointCloudC::Ptr sceneNew,
                                    const PointCloudC::Ptr corrRef,
                                    const PointCloudC::Ptr corrNew,
                                    const float inlrThd,
                                    const float smpRate,
                                    const float inlrRate,
                                    const int maxIter,
                                    Eigen::Matrix4f &transMat,
                                    PointCloudC::Ptr cloudOut);

    void register2ScenesRansac( const PointCloudC::Ptr sceneRef,
                                const PointCloudC::Ptr sceneNew,
                                const PointCloudC::Ptr corrRef,
                                const PointCloudC::Ptr corrNew,
                                const float inlrThd,
                                const int sampleNb,
                                const float inlrRate,
                                const int maxIter,
                                Eigen::Matrix4f &transMat,
                                PointCloudC::Ptr cloudOut);

    void getTransformation3d( const PointCloudC::Ptr corrRef,
                              const PointCloudC::Ptr corrNew,
                              const std::vector<int> inlrIdx,
                              Eigen::Matrix4f &transMat);

    void getTransformation3d( const PointCloudC::Ptr corrRef,
                              const PointCloudC::Ptr corrNew,
                              const std::set<int> sampleIdx,
                              const float inlrThd,
                              Eigen::Matrix4f &transMat,
                              std::vector<float>& transDist,
                              std::vector<int>& inlrIdx,
                              int &inlrNb);
    void getTransform3dGeometric(const PointCloudC::Ptr corrRef,
                                 const PointCloudC::Ptr corrNew,
                                 const std::set<int> sampleIdx,
                                 const float inlrThd,
                                 Eigen::Matrix4f &transMat,
                                 std::vector<float>& transDist,
                                 std::vector<int>& inlrIdx,
                                 int &inlrNb);

    void normalizePointClouds(PointCloudC::Ptr &corrRef,
                              Eigen::Matrix4f &transMat);
    void getRTGeometricLinearSystem(PointCloudC::Ptr corrRef,
                                    PointCloudC::Ptr corrNew,
                                    Eigen::Matrix4f &transMat);
    void randIdx(int idxRange, int sampleNb, std::time_t t, std::set<int>& randSmp);
    void getMaxIdx(const std::vector<int> vec, int &maxIdx);
};

#endif // POINTCLOUDPROCESSING_H
