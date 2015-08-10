#include "pointcloudprocessing.h"
#include <pcl/common/centroid.h>
#include <Eigen/Dense>

//#include <random>
//#include <algorithm>
//#include <cassert>
PointCloudProcessing::PointCloudProcessing()
{
}

void PointCloudProcessing::getKnnRadius(const PointCloudT::Ptr &cloud,
                                        const PointCloudT::Ptr &ptQuery,
                                        const f32 searchRadius,
                                        std::vector< std::vector<s16> > &neighIdx,
                                        std::vector< std::vector<f32> > &neighDist)
{
    //    std::cout<<"search radius: "<<searchRadius<<std::endl;
    pcl::KdTreeFLANN<PointT> kdTree;
    kdTree.setInputCloud(cloud);
    std::vector<s16> knnIdx;
    std::vector<f32> knnDist;

    // create a new file or select the existing files to continue saving selected features
    for(size_t i = 0; i<ptQuery->points.size(); i++)
    {
        PointT pQuery = ptQuery->points.at(i);
        kdTree.radiusSearch(pQuery, searchRadius, knnIdx, knnDist);
        neighIdx.push_back(knnIdx);
        neighDist.push_back(knnDist);
    }
}

void PointCloudProcessing::pclRegionGrow(PointCloudT::Ptr scene,
                                         PointCloudT::Ptr seeds,
                                         float growSpeed,
                                         float searchRadius,
                                         float heightThd,
                                         PointCloudT::Ptr &cloudSeg,
                                         std::set<int> &clusterIdx)
{
    int clusterSize = 0;
    std::vector<std::vector<int> > neighIdx;
    std::vector< std::vector<f32> > neighDist;
    std::set<int> newSeedIdx;
    std::cout<<"seeds size: "<<seeds->points.size()<<std::endl;
    // seeded region growing
    do
    {
        clusterSize = clusterIdx.size(); newSeedIdx.clear();
        //        std::cout<<"seeds size: "<<seeds->points.size()<<std::endl;
        getKnnRadius(scene, seeds, searchRadius, neighIdx, neighDist);
        // update cluster elements
        std::vector<std::vector<int> >::iterator seedNeighIdx = neighIdx.begin();
        std::vector<std::vector<float> >::iterator seedNeighDist = neighDist.begin();
        for(int i=0; i<neighIdx.size(); i++, seedNeighIdx++, seedNeighDist++)
        {
            for(int j=0; j<seedNeighIdx->size(); j++)
            {
                clusterIdx.insert((*seedNeighIdx).at(j));
                if((*seedNeighDist).at(j)>growSpeed*searchRadius)
                {  newSeedIdx.insert((*seedNeighIdx).at(j));  }
            }
        }
        //        std::cout<<"clusterIdx size: "<<clusterIdx.size()<<std::endl;
        // newSeeds for next growing iteration
        PointCloudT::Ptr newSeeds(new PointCloudT); newSeeds->points.clear();
        std::set<int>::iterator newSeedIdxIter = newSeedIdx.begin();
        for(int i=0; i<newSeedIdx.size(); i++, newSeedIdxIter++)
        {
            PointT newSeed = scene->points.at(*newSeedIdxIter);
            if(newSeed.z > heightThd)
            {
                newSeeds->points.push_back(newSeed);
            }
        }
        seeds->points.clear();
        pcl::copyPointCloud(*newSeeds, *seeds);
        //        std::cout<<"newSeeds size: "<<newSeeds->points.size()<<std::endl;
        //        std::cout<<"clusterSize: "<<clusterSize<<"\n";
    }while(clusterSize != clusterIdx.size());

    // get segmented point cloud
    cloudSeg.reset(new PointCloudT); cloudSeg->points.clear();
    std::set<int>::iterator clusterIdxIter = clusterIdx.begin();
    for(int i=0; i<clusterIdx.size(); i++, clusterIdxIter++)
    {
        PointT pointSeg = scene->points.at(*clusterIdxIter);
        if(pointSeg.z > heightThd)
        {
            cloudSeg->points.push_back(pointSeg);
        }
    }
    std::cout<<"cloudSeg size: "<<cloudSeg->points.size()<<std::endl;
}


//////////////////////////////////////////////////////////////
/// 3D registration from point correspondences with Ransac
//////////////////////////////////////////////////////////////
void PointCloudProcessing::register2ScenesMEstimator(
        const PointCloudT::Ptr sceneRef,
        const PointCloudT::Ptr sceneNew,
        const PointCloudT::Ptr corrRef,
        const PointCloudT::Ptr corrNew,
        const float inlrThd,
        const float smpRate,
        const float inlrRate,
        const int maxIter,
        Eigen::Matrix4f &transMat,
        PointCloudT::Ptr cloudOut)
{
    int corrSize = corrRef->points.size();
    std::cout<<"corrSize: "<<corrSize<<std::endl;
    int sampleNb = smpRate*corrSize;
    std::cout<<"sampleNb: "<<sampleNb<<std::endl;
    std::cout<<"inlrThd: "<<sampleNb*inlrRate<<std::endl;
    std::cout<<"maxIter: "<<maxIter<<std::endl;
    std::vector<int> inlrNbAll;
    std::vector<Eigen::Matrix4f> transMatAll;
    int iter = 0;
    std::time_t timeRnd;
    std::time(&timeRnd);
    while (iter<maxIter)
    {
        std::cout<<"iter = "<<iter<<"\n\n";
        int inlrNb = 0;
        // Random sampling
        std::set<int> sampleIdx; sampleIdx.clear();
        std::cout<<"sampleIdx size: "<<sampleIdx.size()<<std::endl;
        randIdx(corrSize, sampleNb, timeRnd, sampleIdx);
        // 1. Transform point cloud and compute distance;
        std::vector<float> transDist;
        std::vector<int> inlrIdx;
        getTransformation3d(corrRef, corrNew, sampleIdx, inlrThd,\
                            transMat, transDist, inlrIdx, inlrNb);
        std::cout<<"inlrIdx size"<<inlrIdx.size()<<"\n";
        std::cout<<"get transform iter done...\n";
        // 2. count the inlrs, if more than thInlr
        if(inlrNb < sampleNb*inlrRate)
        {
            ++iter; continue;
        }
        std::cout<<"inlrIdx size: "<<inlrIdx.size()<<"\n";
        getTransformation3d(corrRef, corrNew, inlrIdx, \
                            transMat);
        inlrNbAll.push_back(inlrNb);
        transMatAll.push_back(transMat);
        ++iter;
        std::cout<<"inlrNb: "<<inlrNb<<"\t";
    }
    // 3. choose the coef with the most inliers
    int bestSmp = 0;
    getMaxIdx(inlrNbAll, bestSmp);
    std::cout<<"best sample idx: "<<bestSmp<<"\n";
    transMat = transMatAll.at(bestSmp);
    std::cout<<"transMat: "<<transMat<<std::endl;
    // Transform point cloud
    pcl::transformPointCloud(*sceneNew, *sceneNew, transMat);
    pcl::copyPointCloud(*sceneRef,*cloudOut);
    for(int i=0; i<sceneNew->points.size();i++)
    {
        cloudOut->points.push_back(sceneNew->points.at(i));
    }
}

void PointCloudProcessing::register2ScenesRansac(
        const PointCloudT::Ptr sceneRef,
        const PointCloudT::Ptr sceneNew,
        const PointCloudT::Ptr corrRef,
        const PointCloudT::Ptr corrNew,
        const float inlrThd,
        const int sampleNb,
        const float inlrRate,
        const int maxIter,
        Eigen::Matrix4f &transMat,
        PointCloudT::Ptr cloudOut)
{
    int corrSize = corrRef->points.size();
    std::cout<<"corrSize: "<<corrSize<<std::endl;
    std::cout<<"inlrThd: "<<sampleNb*inlrRate<<std::endl;
    std::cout<<"maxIter: "<<maxIter<<std::endl;
    std::vector<int> inlrNbAll;
    std::vector<Eigen::Matrix4f> transMatAll;
    int iter = 0;
    std::time_t timeRnd;
    std::time(&timeRnd);
    while (iter<maxIter)
    {
        std::cout<<"iter = "<<iter<<"\n\n";
        int inlrNb = 0;
        // Random sampling
        std::set<int> sampleIdx; sampleIdx.clear();
        std::cout<<"sampleIdx size: "<<sampleIdx.size()<<std::endl;
        randIdx(corrSize, sampleNb, timeRnd, sampleIdx);
        // 1. Transform point cloud and compute distance;
        std::vector<float> transDist;
        std::vector<int> inlrIdx;
        getTransformation3d(corrRef, corrNew, sampleIdx, inlrThd,\
                            transMat, transDist, inlrIdx, inlrNb);
        std::cout<<"inlrIdx size"<<inlrIdx.size()<<"\n";
        std::cout<<"get transform iter done...\n";
        // 2. count the inlrs, if more than thInlr
        if(inlrNb < sampleNb*inlrRate)
        {
            ++iter; continue;
        }
        std::cout<<"inlrIdx size: "<<inlrIdx.size()<<"\n";
        getTransformation3d(corrRef, corrNew, inlrIdx, \
                            transMat);
        inlrNbAll.push_back(inlrNb);
        transMatAll.push_back(transMat);
        ++iter;
        std::cout<<"inlrNb: "<<inlrNb<<"\t";
    }
    // 3. choose the coef with the most inliers
    int bestSmp = 0;
    getMaxIdx(inlrNbAll, bestSmp);
    std::cout<<"best sample idx: "<<bestSmp<<"\n";
    transMat = transMatAll.at(bestSmp);
    std::cout<<"transMat: "<<transMat<<std::endl;
    // Transform point cloud
    pcl::transformPointCloud(*sceneNew, *sceneNew, transMat);
    pcl::copyPointCloud(*sceneRef,*cloudOut);
    for(int i=0; i<sceneNew->points.size();i++)
    {
        cloudOut->points.push_back(sceneNew->points.at(i));
    }
}


void PointCloudProcessing::getTransformation3d(
        const PointCloudT::Ptr corrRef,
        const PointCloudT::Ptr corrNew,
        const std::vector<int> inlrIdx,
        Eigen::Matrix4f &transMat)
{
    // 3D rigid transformation estimation using SVD
    pcl::TransformationFromCorrespondences transFromCorr;
    for(int i=0; i<inlrIdx.size(); i++)
    {
        PointT pointRef = corrRef->points.at(inlrIdx.at(i));
        PointT pointNew = corrNew->points.at(inlrIdx.at(i));
        Eigen::Vector3f from( pointNew.x, pointNew.y, pointNew.z);
        Eigen::Vector3f  to ( pointRef.x, pointRef.y, pointRef.z);
        transFromCorr.add(from, to, 1.0);//all the same weight
    }
    // Get transformation matrix
    transMat= transFromCorr.getTransformation().matrix();
    std::cout<<"transMat computed...\n";
}

void PointCloudProcessing::getTransform3dGeometric(
        const PointCloudT::Ptr corrRef,
        const PointCloudT::Ptr corrNew,
        const std::set<int> sampleIdx,
        const float inlrThd,
        Eigen::Matrix4f &transMat,
        std::vector<float>& transDist,
        std::vector<int>& inlrIdx,
        int &inlrNb)
{
    // 3D rigid transformation estimation using SVD
    pcl::TransformationFromCorrespondences transFromCorr;
    PointCloudT::Ptr smpRef(new PointCloudT), smpNew(new PointCloudT);
    std::set<int>::iterator setIter = sampleIdx.begin();
    std::cout<<"start transMat computation...\n";
    while(setIter!=sampleIdx.end())
    {
//        std::cout<<"smpIdx: "<<*setIter<<", ";
        PointT pointRef = corrRef->points.at(*setIter);
        smpRef->push_back(pointRef);
        PointT pointNew = corrNew->points.at(*setIter);
        smpNew->push_back(pointNew);
        Eigen::Vector3f from( pointNew.x, pointNew.y, pointNew.z);
        Eigen::Vector3f  to ( pointRef.x, pointRef.y, pointRef.z);
        transFromCorr.add(from, to, 1.0);//all the same weight
        ++setIter;
    }
    // Get transformation matrix
    transMat= transFromCorr.getTransformation().matrix();
    std::cout<<"iteration transMat: "<<transMat<<"\n";

    // Get transformed point distance
    pcl::transformPointCloud(*smpNew, *smpNew, transMat);
    setIter = sampleIdx.begin();
    std::cout<<"Get transformed point distance started ...\n";
    for(int i=0; i<smpRef->points.size();i++, setIter++)
    {
        PointT ptRef = smpRef->points.at(i);
        PointT ptNew = smpNew->points.at(i);

        float ptDist = std::sqrt((ptRef.x-ptNew.x)*(ptRef.x-ptNew.x) + \
                                 (ptRef.y-ptNew.y)*(ptRef.y-ptNew.y) + \
                                 (ptRef.z-ptNew.z)*(ptRef.z-ptNew.z) );
        transDist.push_back(ptDist);
        if(i<20){std::cout<<"ptDist: "<<ptDist<<"\t";}
        if(ptDist<inlrThd)
        {
            ++inlrNb;
            inlrIdx.push_back(*setIter);
        }
    }
    std::cout<<"Get transformed point distance done...\n";
}
void PointCloudProcessing::getRTGeometricLinearSystem(PointCloudT::Ptr corrRef,
                                                      PointCloudT::Ptr corrNew,
                                                      Eigen::Matrix4f &transMat)
{
    // build linear system Ax = b;
    int rows = 3*corrRef->points.size();
    int cols = 6;
    Eigen::MatrixXf A(rows, cols); A.setZero();
    Eigen::MatrixXf b(rows, 1); b.setZero();
    for(int i=0; i<corrRef->points.size(); i++)
    {
        float X1 = corrRef->points.at(i).x;
        float Y1 = corrRef->points.at(i).y;
        float Z1 = corrRef->points.at(i).z;
        float X0 = corrNew->points.at(i).x;
        float Y0 = corrNew->points.at(i).y;
        float Z0 = corrNew->points.at(i).z;
        A(3*i, 0) = 0;      A(3*i, 1) = Z0+Z1;  A(3*i, 2) = -Y0-Y1; A(3*i, 3) = 1;
        A(3*i, 0) = -Z0-Z1; A(3*i, 1) = 0;      A(3*i, 2) = X0+X1;  A(3*i, 4) = 1;
        A(3*i, 0) = Y0+Y1;  A(3*i, 1) = -X0-X1; A(3*i, 2) = 0;      A(3*i, 5) = 1;

        b(3*i,   0) = X1-X0;
        b(3*i+1, 0) = Y1-Y0;
        b(3*i+2, 0) = Z1-Z0;
    }
    Eigen::MatrixXf solveX(rows, 1); solveX.setZero();
    solveX = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
}
void PointCloudProcessing::getTransformation3d(
        const PointCloudT::Ptr corrRef,
        const PointCloudT::Ptr corrNew,
        const std::set<int> sampleIdx,
        const float inlrThd,
        Eigen::Matrix4f &transMat,
        std::vector<float>& transDist,
        std::vector<int>& inlrIdx,
        int &inlrNb)
{
    // 3D rigid transformation estimation using SVD
    pcl::TransformationFromCorrespondences transFromCorr;
    PointCloudT::Ptr smpRef(new PointCloudT), smpNew(new PointCloudT);
    std::set<int>::iterator setIter = sampleIdx.begin();
    std::cout<<"start transMat computation...\n";
    while(setIter!=sampleIdx.end())
    {
//        std::cout<<"smpIdx: "<<*setIter<<", ";
        PointT pointRef = corrRef->points.at(*setIter);
        smpRef->push_back(pointRef);
        PointT pointNew = corrNew->points.at(*setIter);
        smpNew->push_back(pointNew);
        Eigen::Vector3f from( pointNew.x, pointNew.y, pointNew.z);
        Eigen::Vector3f  to ( pointRef.x, pointRef.y, pointRef.z);
        transFromCorr.add(from, to, 1.0);//all the same weight
        ++setIter;
    }
    // Get transformation matrix
    transMat= transFromCorr.getTransformation().matrix();
    std::cout<<"iteration transMat: "<<transMat<<"\n";

    // Get transformed point distance
    pcl::transformPointCloud(*smpNew, *smpNew, transMat);
    setIter = sampleIdx.begin();
    std::cout<<"Get transformed point distance started ...\n";
    for(int i=0; i<smpRef->points.size();i++, setIter++)
    {
        PointT ptRef = smpRef->points.at(i);
        PointT ptNew = smpNew->points.at(i);

        float ptDist = std::sqrt((ptRef.x-ptNew.x)*(ptRef.x-ptNew.x) + \
                                 (ptRef.y-ptNew.y)*(ptRef.y-ptNew.y) + \
                                 (ptRef.z-ptNew.z)*(ptRef.z-ptNew.z) );
        transDist.push_back(ptDist);
        if(i<20){std::cout<<"ptDist: "<<ptDist<<"\t";}
        if(ptDist<inlrThd)
        {
            ++inlrNb;
            inlrIdx.push_back(*setIter);
        }
    }
    std::cout<<"Get transformed point distance done...\n";
}

void PointCloudProcessing::randIdx(int idxRange, int sampleNb, std::time_t t,
                                   std::set<int>& randSmp)
{
    std::srand((unsigned int) (t*(std::rand()%1000)));
    while(randSmp.size()<sampleNb)
    {
        int randNum = std::rand() % idxRange;
        randSmp.insert(randNum);
    }
}


void PointCloudProcessing::getMaxIdx(const std::vector<int> vec, int maxIdx)
{
    int maxInlrNb = 0;
    for(int i=0; i<vec.size(); i++)
    {
        if(vec.at(i)>maxInlrNb)
        {
            maxIdx = i;
        }
    }
}

void PointCloudProcessing::normalizePointClouds(PointCloudT::Ptr &corrRef,
                                                PointCloudT::Ptr &corrNew,
                                                Eigen::Matrix4f &transMat)
{
    // shift cloud to centroid
    PointT centroidXYZ;
    pcl::computeCentroid(*corrRef, centroidXYZ);
    transMat << 1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0;
    transMat(0,3) = centroidXYZ.x;
    transMat(1,3) = centroidXYZ.y;
    transMat(2,3) = centroidXYZ.z;
    // get normalization scale
    double distScl = 0.0;
    for(int i=0; i<corrRef->points.size(); i++)
    {
        PointT pt = corrRef->points.at(i);
        distScl += std::sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z);
    }
    distScl = distScl/corrRef->points.size();
    distScl = sqrt(3)/distScl;
    transMat(0,0) = distScl;
    transMat(1,1) = distScl;
    transMat(2,2) = distScl;
//    std::cout<<"transMat normalization: "<<transMat;
    pcl::transformPointCloud(*corrRef, *corrRef, transMat);
    pcl::transformPointCloud(*corrNew, *corrNew, transMat);
}
