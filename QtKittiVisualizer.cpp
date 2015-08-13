#include "QtKittiVisualizer.h"
#include "ui_kittiQviewer.h"
#include <fstream>
#include <string>
#include <iostream>

#include <QCheckBox>
#include <QLabel>
#include <QMainWindow>
#include <QSlider>
#include <QWidget>
#include <QInputDialog>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <eigen3/Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/LU>

#include <pcl/common/transforms.h>
#include <pcl/filters/crop_box.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/sample_consensus/mlesac.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <flann/flann.h>
#include <boost/filesystem.hpp>


#include <KittiConfig.h>
#include <KittiDataset.h>

#include <kitti-devkit-raw/tracklets.h>

typedef pcl::visualization::PointCloudColorHandlerCustom<KittiPoint> KittiPointCloudColorHandlerCustom;

KittiVisualizerQt::KittiVisualizerQt(QWidget *parent, int argc, char** argv) :
    QMainWindow(parent),
    ui(new Ui::KittiVisualizerQt),
    dataset_index(0),
    frame_index(0),
    tracklet_index(0),
    pclVisualizer(new pcl::visualization::PCLVisualizer("PCL Visualizer", false)),
    pointCloudVisible(true),
    trackletBoundingBoxesVisible(true),
    trackletPointsVisible(true),
    trackletInCenterVisible(true)
{
    int invalidOptions = parseCommandLineOptions(argc, argv);
    if (invalidOptions)
    {
        exit(invalidOptions);
    }

    // Set up user interface
    ui->setupUi(this);
    ui->qvtkWidget_pclViewer->SetRenderWindow(pclVisualizer->getRenderWindow());
    pclVisualizer->setupInteractor(ui->qvtkWidget_pclViewer->GetInteractor(), ui->qvtkWidget_pclViewer->GetRenderWindow());
    pclVisualizer->setBackgroundColor(0, 0, 0);
    pclVisualizer->addCoordinateSystem(1.0);
    pclVisualizer->registerKeyboardCallback(&KittiVisualizerQt::keyboardEventOccurred, *this, 0);
    this->setWindowTitle("Qt KITTI Visualizer");
    ui->qvtkWidget_pclViewer->update();

    // Init the viewer with the first point cloud and corresponding tracklets
    dataset = new KittiDataset(KittiConfig::availableDatasets.at(dataset_index));
    loadPointCloud();
    if (pointCloudVisible)
        showPointCloud();
    loadAvailableTracklets();
    if (trackletBoundingBoxesVisible)
        showTrackletBoxes();
    loadTrackletPoints();
    if (trackletPointsVisible)
        showTrackletPoints();
    if (trackletInCenterVisible)
        showTrackletInCenter();

    ui->slider_dataSet->setRange(0, KittiConfig::availableDatasets.size() - 1);
    ui->slider_dataSet->setValue(dataset_index);
    ui->slider_frame->setRange(0, dataset->getNumberOfFrames() - 1);
    ui->slider_frame->setValue(frame_index);
    if (availableTracklets.size() != 0)
        ui->slider_tracklet->setRange(0, availableTracklets.size() - 1);
    else
        ui->slider_tracklet->setRange(0, 0);
    ui->slider_tracklet->setValue(tracklet_index);

    updateDatasetLabel();
    updateFrameLabel();
    updateTrackletLabel();

    // Connect signals and slots
    connect(ui->slider_dataSet,  SIGNAL (valueChanged(int)), this, SLOT (newDatasetRequested(int)));
    connect(ui->slider_frame,    SIGNAL (valueChanged(int)), this, SLOT (newFrameRequested(int)));
    connect(ui->slider_tracklet, SIGNAL (valueChanged(int)), this, SLOT (newTrackletRequested(int)));
    connect(ui->checkBox_showFramePointCloud,       SIGNAL (toggled(bool)), this, SLOT (showFramePointCloudToggled(bool)));
    connect(ui->checkBox_showTrackletBoundingBoxes, SIGNAL (toggled(bool)), this, SLOT (showTrackletBoundingBoxesToggled(bool)));
    connect(ui->checkBox_showTrackletPointClouds,   SIGNAL (toggled(bool)), this, SLOT (showTrackletPointCloudsToggled(bool)));
    connect(ui->checkBox_showTrackletInCenter,      SIGNAL (toggled(bool)), this, SLOT (showTrackletInCenterToggled(bool)));

    colorOpt[0] = 0;
    colorOpt[1] = 0;
    colorOpt[2] = 0;
    pointCloud.reset(new KittiPointCloud);
    colorCloud.reset(new PointCloudC);
    colorTrk.reset(new PointCloudC);
    scene.reset(new PointCloudC);
    seeds.reset(new PointCloudC);
    searchRadius = 0.8;
    heightThd = -1.5;
    growSpeed = 0.5;
    clusterIdx.clear();

    // 2 scenes registration data
    str2SceneRansacParams = new str_2ScenesRansac;
    str2SceneRansacParams->sceneRef.reset(new PointCloudC);
    str2SceneRansacParams->sceneNew.reset(new PointCloudC);
    str2SceneRansacParams->corrRef .reset(new PointCloudC);
    str2SceneRansacParams->corrNew .reset(new PointCloudC);
    str2SceneRansacParams->motSeedsRef.reset(new PointCloudC);
    str2SceneRansacParams->motSeedsNew.reset(new PointCloudC);
    str2SceneRansacParams->sceneMotRef  .reset(new PointCloudC);
    str2SceneRansacParams->sceneMotNew  .reset(new PointCloudC);
    str2SceneRansacParams->sceneNoMotRef.reset(new PointCloudC);
    str2SceneRansacParams->sceneNoMotNew.reset(new PointCloudC);
    str2SceneRansacParams->registeredScene.reset(new PointCloudC);
    str2SceneRansacParams->mergedScene.reset(new PointCloudC);

    // point cloud display names
    str2SceneRansacParams->sceneRefName    = "load Reference Scene";
    str2SceneRansacParams->sceneNewName    = "load New Scene";
    str2SceneRansacParams->corrRefName     = "load Reference Correspondences";
    str2SceneRansacParams->corrNewName     = "load New Correspondences";
    str2SceneRansacParams->motSeedsRefName = "load Reference Motion Seeds";
    str2SceneRansacParams->motSeedsNewName = "load New Motion Seeds";
    str2SceneRansacParams->registeredName  = "registeredCloud";

    // Ransac parameter settings
    str2SceneRansacParams->inlrThdRansac = 0.05;
    str2SceneRansacParams->smpRateRansac = 0.7;
    str2SceneRansacParams->inlrRateRansac = 0.8;
    str2SceneRansacParams->maxIterRansac = 10000;
    str2SceneRansacParams->sampleNbRansac = 3;
    str2SceneRansacParams->registLength = 1;
    str2SceneRansacParams->motIdxRef.clear();
    str2SceneRansacParams->motIdxNew.clear();

    // Initialize transformation matrix
    str2SceneRansacParams->transMat <<  1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0;

    str2SceneRansacParams->normalizationMat <<  1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0;
}

KittiVisualizerQt::~KittiVisualizerQt()
{
    delete dataset;
    delete ui;
}

int KittiVisualizerQt::parseCommandLineOptions(int argc, char** argv)
{
    // Declare the supported options.
    boost::program_options::options_description desc("Program options");
    desc.add_options()
            ("help", "Produce this help message.")
            ("dataset", boost::program_options::value<int>(), "Set the number of the KITTI data set to be used.")
            ;

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
    }

    if (vm.count("dataset")) {
        dataset_index = vm["dataset"].as<int>();
        std::cout << "Using data set " << dataset_index << "." << std::endl;
        dataset_index = KittiConfig::getDatasetIndex(dataset_index);
    } else {
        dataset_index = 0;
        std::cout << "Data set was not specified." << std::endl;
        std::cout << "Using data set " << KittiConfig::getDatasetNumber(dataset_index) << "." << std::endl;
    }
    return 0;
}

bool KittiVisualizerQt::loadNextFrame()
{
    newFrameRequested(frame_index + 1);
}

bool KittiVisualizerQt::loadPreviousFrame()
{
    newFrameRequested(frame_index - 1);
}

void KittiVisualizerQt::getTrackletColor(const KittiTracklet& tracklet, int &r, int& g, int& b)
{
    KittiDataset::getColor(tracklet.objectType.c_str(), r, g, b);
}

void KittiVisualizerQt::showFramePointCloudToggled(bool value)
{
    pointCloudVisible = value;
    if (pointCloudVisible)
    {
        showPointCloud();
    }
    else
    {
        hidePointCloud();
    }
    ui->qvtkWidget_pclViewer->update();
}

void KittiVisualizerQt::newDatasetRequested(int value)
{
    if (dataset_index == value)
        return;

    if (trackletInCenterVisible)
        hideTrackletInCenter();
    if (trackletPointsVisible)
        hideTrackletPoints();
    clearTrackletPoints();
    if (trackletBoundingBoxesVisible)
        hideTrackletBoxes();
    clearAvailableTracklets();
    if (pointCloudVisible)
        hidePointCloud();

    dataset_index = value;
    if (dataset_index >= KittiConfig::availableDatasets.size())
        dataset_index = KittiConfig::availableDatasets.size() - 1;
    if (dataset_index < 0)
        dataset_index = 0;

    delete dataset;
    dataset = new KittiDataset(KittiConfig::availableDatasets.at(dataset_index));

    if (frame_index >= dataset->getNumberOfFrames())
        frame_index = dataset->getNumberOfFrames() - 1;

    loadPointCloud();
    if (pointCloudVisible)
        showPointCloud();
    loadAvailableTracklets();
    if (trackletBoundingBoxesVisible)
        showTrackletBoxes();
    loadTrackletPoints();
    if (trackletPointsVisible)
        showTrackletPoints();
    if (tracklet_index >= availableTracklets.size())
        tracklet_index = availableTracklets.size() - 1;
    if (tracklet_index < 0)
        tracklet_index = 0;
    if (trackletInCenterVisible)
        showTrackletInCenter();

    ui->slider_frame->setRange(0, dataset->getNumberOfFrames() - 1);
    ui->slider_frame->setValue(frame_index);
    if (availableTracklets.size() != 0)
        ui->slider_tracklet->setRange(0, availableTracklets.size() - 1);
    else
        ui->slider_tracklet->setRange(0, 0);
    ui->slider_tracklet->setValue(tracklet_index);

    updateDatasetLabel();
    updateFrameLabel();
    updateTrackletLabel();
    ui->qvtkWidget_pclViewer->update();
}

void KittiVisualizerQt::newFrameRequested(int value)
{
    if (frame_index == value)
        return;

    if (trackletInCenterVisible)
        hideTrackletInCenter();
    if (trackletPointsVisible)
        hideTrackletPoints();
    clearTrackletPoints();
    if (trackletBoundingBoxesVisible)
        hideTrackletBoxes();
    clearAvailableTracklets();
    if (pointCloudVisible)
        hidePointCloud();

    frame_index = value;
    if (frame_index >= dataset->getNumberOfFrames())
        frame_index = dataset->getNumberOfFrames() - 1;
    if (frame_index < 0)
        frame_index = 0;

    loadPointCloud();
    if (pointCloudVisible)
        showPointCloud();
    loadAvailableTracklets();
    if (trackletBoundingBoxesVisible)
        showTrackletBoxes();
    loadTrackletPoints();
    if (trackletPointsVisible)
        showTrackletPoints();
    if (tracklet_index >= availableTracklets.size())
        tracklet_index = availableTracklets.size() - 1;
    if (tracklet_index < 0)
        tracklet_index = 0;
    if (trackletInCenterVisible)
        showTrackletInCenter();

    if (availableTracklets.size() != 0)
        ui->slider_tracklet->setRange(0, availableTracklets.size() - 1);
    else
        ui->slider_tracklet->setRange(0, 0);
    ui->slider_tracklet->setValue(tracklet_index);

    updateFrameLabel();
    updateTrackletLabel();
    ui->qvtkWidget_pclViewer->update();
}

void KittiVisualizerQt::newTrackletRequested(int value)
{
    if (tracklet_index == value)
        return;

    if (trackletInCenterVisible)
        hideTrackletInCenter();

    tracklet_index = value;
    if (tracklet_index >= availableTracklets.size())
        tracklet_index = availableTracklets.size() - 1;
    if (tracklet_index < 0)
        tracklet_index = 0;
    if (trackletInCenterVisible)
        showTrackletInCenter();

    updateTrackletLabel();
    ui->qvtkWidget_pclViewer->update();
}

void KittiVisualizerQt::showTrackletBoundingBoxesToggled(bool value)
{
    trackletBoundingBoxesVisible = value;
    if (trackletBoundingBoxesVisible)
    {
        showTrackletBoxes();
    }
    else
    {
        hideTrackletBoxes();
    }
    ui->qvtkWidget_pclViewer->update();
}

void KittiVisualizerQt::showTrackletPointCloudsToggled(bool value)
{
    trackletPointsVisible = value;
    if (trackletPointsVisible)
    {
        showTrackletPoints();
    }
    else
    {
        hideTrackletPoints();
    }
    ui->qvtkWidget_pclViewer->update();
}

void KittiVisualizerQt::showTrackletInCenterToggled(bool value)
{
    trackletInCenterVisible = value;
    if (trackletInCenterVisible)
    {
        showTrackletInCenter();
    }
    else
    {
        hideTrackletInCenter();
    }
    ui->qvtkWidget_pclViewer->update();
}

void KittiVisualizerQt::loadPointCloud()
{
    pointCloud = dataset->getPointCloud(frame_index);
}

void KittiVisualizerQt::showPointCloud()
{
    KittiPointCloudColorHandlerCustom colorHandler(pointCloud, 255, 255, 255);
    pclVisualizer->addPointCloud<KittiPoint>(pointCloud, colorHandler, "point_cloud");
}

void KittiVisualizerQt::hidePointCloud()
{
    pclVisualizer->removePointCloud("point_cloud");
}

void KittiVisualizerQt::loadAvailableTracklets()
{
    Tracklets& tracklets = dataset->getTracklets();
    int tracklet_id;
    int number_of_tracklets = tracklets.numberOfTracklets();
    for (tracklet_id = 0; tracklet_id < number_of_tracklets; tracklet_id++)
    {
        KittiTracklet* tracklet = tracklets.getTracklet(tracklet_id);
        if (tracklet->first_frame <= frame_index && tracklet->lastFrame() >= frame_index)
        {
            availableTracklets.push_back(*tracklet);
        }
    }
}

void KittiVisualizerQt::clearAvailableTracklets()
{
    availableTracklets.clear();
}

void KittiVisualizerQt::updateDatasetLabel()
{
    std::stringstream text;
    text << "Data set: "
         << dataset_index + 1 << " of " << KittiConfig::availableDatasets.size()
         << " [" << KittiConfig::getDatasetNumber(dataset_index) << "]"
         << std::endl;
    ui->label_dataSet->setText(text.str().c_str());
}

void KittiVisualizerQt::updateFrameLabel()
{
    std::stringstream text;
    text << "Frame: "
         << frame_index + 1 << " of " << dataset->getNumberOfFrames()
         << std::endl;
    ui->label_frame->setText(text.str().c_str());

}

void KittiVisualizerQt::updateTrackletLabel()
{
    if (availableTracklets.size())
    {
        std::stringstream text;
        KittiTracklet tracklet = availableTracklets.at(tracklet_index);
        text << "Tracklet: "
             << tracklet_index + 1 << " of " << availableTracklets.size()
             << " (\"" << tracklet.objectType
             << "\", " << croppedTrackletPointClouds.at(tracklet_index).get()->size()
             << " points)"
             << std::endl;
        ui->label_tracklet->setText(text.str().c_str());
    }
    else
    {
        std::stringstream text;
        text << "Tracklet: "
             << "0 of 0"
             << std::endl;
        ui->label_tracklet->setText(text.str().c_str());
    }
}

void KittiVisualizerQt::showTrackletBoxes()
{
    double boxHeight = 0.0d;
    double boxWidth = 0.0d;
    double boxLength = 0.0d;
    int pose_number = 0;

    for (int i = 0; i < availableTracklets.size(); ++i)
    {
        // Create the bounding box
        const KittiTracklet& tracklet = availableTracklets.at(i);

        boxHeight = tracklet.h;
        boxWidth = tracklet.w;
        boxLength = tracklet.l;
        pose_number = frame_index - tracklet.first_frame;
        const Tracklets::tPose& tpose = tracklet.poses.at(pose_number);
        Eigen::Vector3f boxTranslation;
        boxTranslation[0] = (float) tpose.tx;
        boxTranslation[1] = (float) tpose.ty;
        boxTranslation[2] = (float) tpose.tz + (float) boxHeight / 2.0f;
        Eigen::Quaternionf boxRotation = Eigen::Quaternionf(Eigen::AngleAxisf((float) tpose.rz, Eigen::Vector3f::UnitZ()));

        // Add the bounding box to the visualizer
        std::string viewer_id = "tracklet_box_" + i;
        pclVisualizer->addCube(boxTranslation, boxRotation, boxLength, boxWidth, boxHeight, viewer_id);
    }
}

void KittiVisualizerQt::hideTrackletBoxes()
{
    for (int i = 0; i < availableTracklets.size(); ++i)
    {
        std::string viewer_id = "tracklet_box_" + i;
        pclVisualizer->removeShape(viewer_id);
    }
}

void KittiVisualizerQt::loadTrackletPoints()
{
    for (int i = 0; i < availableTracklets.size(); ++i)
    {
        // Create the tracklet point cloud
        const KittiTracklet& tracklet = availableTracklets.at(i);
        pcl::PointCloud<KittiPoint>::Ptr trackletPointCloud = dataset->getTrackletPointCloud(pointCloud, tracklet, frame_index);
        pcl::PointCloud<KittiPoint>::Ptr trackletPointCloudTransformed(new pcl::PointCloud<KittiPoint>);

        Eigen::Vector3f transformOffset;
        transformOffset[0] = 0.0f;
        transformOffset[1] = 0.0f;
        transformOffset[2] = 6.0f;
        pcl::transformPointCloud(*trackletPointCloud, *trackletPointCloudTransformed, transformOffset, Eigen::Quaternionf::Identity());

        // Store the tracklet point cloud
        croppedTrackletPointClouds.push_back(trackletPointCloudTransformed);
    }
}

void KittiVisualizerQt::showTrackletPoints()
{
    for (int i = 0; i < availableTracklets.size(); ++i)
    {
        // Create a color handler for the tracklet point cloud
        const KittiTracklet& tracklet = availableTracklets.at(i);
        int r, g, b;
        getTrackletColor(tracklet, r, g, b);
        KittiPointCloudColorHandlerCustom colorHandler(croppedTrackletPointClouds.at(i), r, g, b);

        // Add tracklet point cloud to the visualizer
        std::string viewer_id = "cropped_tracklet_" + i;
        pclVisualizer->addPointCloud<KittiPoint>(croppedTrackletPointClouds.at(i), colorHandler, viewer_id);
    }
}

void KittiVisualizerQt::hideTrackletPoints()
{
    for (int i = 0; i < availableTracklets.size(); ++i)
    {
        std::string viewer_id = "cropped_tracklet_" + i;
        pclVisualizer->removeShape(viewer_id);
    }
}

void KittiVisualizerQt::clearTrackletPoints()
{
    croppedTrackletPointClouds.clear();
}

void KittiVisualizerQt::showTrackletInCenter()
{
    if (availableTracklets.size())
    {
        // Create the centered tracklet point cloud
        const KittiTracklet& tracklet = availableTracklets.at(tracklet_index);
        pcl::PointCloud<KittiPoint>::Ptr cloudOut = dataset->getTrackletPointCloud(pointCloud, tracklet, frame_index);
        pcl::PointCloud<KittiPoint>::Ptr cloudOutTransformed(new pcl::PointCloud<KittiPoint>);

        int pose_number = frame_index - tracklet.first_frame;
        Tracklets::tPose tpose = tracklet.poses.at(pose_number);

        Eigen::Vector3f transformOffset((float) -(tpose.tx), (float) -(tpose.ty), (float) -(tpose.tz + tracklet.h / 2.0));

        Eigen::AngleAxisf angleAxisZ(-tpose.rz, Eigen::Vector3f::UnitZ());
        Eigen::Quaternionf transformRotation(angleAxisZ);
        pcl::transformPointCloud(*cloudOut, *cloudOutTransformed, transformOffset, Eigen::Quaternionf::Identity());
        pcl::transformPointCloud(*cloudOutTransformed, *cloudOut, Eigen::Vector3f::Zero(), transformRotation);

        // Create color handler for the centered tracklet point cloud
        KittiPointCloudColorHandlerCustom colorHandler(cloudOut, 0, 255, 0);

        // Add the centered tracklet point cloud to the visualizer
        pclVisualizer->addPointCloud<KittiPoint>(cloudOut, colorHandler, "centered_tracklet");
    }
}

void KittiVisualizerQt::hideTrackletInCenter()
{
    if (availableTracklets.size())
    {
        pclVisualizer->removeShape("centered_tracklet");
    }
}

void KittiVisualizerQt::setFrameNumber(int frameNumber)
{
    frame_index = frameNumber;
}

void KittiVisualizerQt::keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event,
                                               void* viewer_void)
{
    if (event.getKeyCode() == 0 && event.keyDown())
    {
        if (event.getKeySym() == "Left")
        {
            loadPreviousFrame();
        }
        else if (event.getKeySym() == "Right")
        {
            loadNextFrame();
        }
    }
}

void getHeatMapColor(float value, float &red, float &green, float &blue)
{
    const int NUM_COLORS = 4;
    static float color[NUM_COLORS][3] = { {0,0,255}, {0,255,0}, {255,255,0}, {255,0,0} };
    // A static array of 4 colors:  (blue,   green,  yellow,  red) using {r,g,b} for each.

    int idx1;        // |-- Our desired color will be between these two indexes in "color".
    int idx2;        // |
    float fractBetween = 0;  // Fraction between "idx1" and "idx2" where our value is.

    if(value <= 0)      {  idx1 = idx2 = 0;            }    // accounts for an input <=0
    else if(value >= 1)  {  idx1 = idx2 = NUM_COLORS-1; }    // accounts for an input >=0
    else
    {
        value = value * (NUM_COLORS-1);        // Will multiply value by 3.
        idx1  = floor(value);                  // Our desired color will be after this index.
        idx2  = idx1+1;                        // ... and before this index (inclusive).
        fractBetween = value - float(idx1);    // Distance between the two indexes (0-1).
    }

    red   = (color[idx2][0] - color[idx1][0])*fractBetween + color[idx1][0];
    green = (color[idx2][1] - color[idx1][1])*fractBetween + color[idx1][1];
    blue  = (color[idx2][2] - color[idx1][2])*fractBetween + color[idx1][2];
}
void getValueBetweenTwoFixedColors(float value, int &red, int &green, int &blue)
{
    int aR = 0;   int aG = 0; int aB=255;  // RGB for our 1st color (blue in this case).
    int bR = 255; int bG = 0; int bB=0;    // RGB for our 2nd color (red in this case).

    red   = (float)(bR - aR) * value + aR;      // Evaluated as -255*value + 255.
    green = (float)(bG - aG) * value + aG;      // Evaluates as 0.
    blue  = (float)(bB - aB) * value + aB;      // Evaluates as 255*value + 0.
}




void KittiVisualizerQt::on_loadPc_clicked()
{
    //    KittiPointCloud::Ptr cloud(new KittiPointCloud);
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"),
                                                    "/home/jiang/CvDataset/KITTI/tracking_module/training/velodyne", \
                                                    tr("Files (*.bin)"));
    std::fstream file(fileName.toStdString().c_str(), std::ios::in | std::ios::binary);
    if(file.good()){
        pointCloud->clear();
        file.seekg(0, std::ios::beg);
        int i;
        for (i = 0; file.good() && !file.eof(); i++) {
            KittiPoint point;
            file.read((char *) &point.x, 3*sizeof(float));
            file.read((char *) &point.intensity, sizeof(float));
            pointCloud->push_back(point);
        }
        file.close();
    }
    for(int i=0; i<pointCloud->points.size(); i++)
    {
        KittiPoint *point = &pointCloud->points.at(i);
        if( (*point).z <-2.5)
        {
            (*point).x = 0; (*point).y = 0; (*point).z = 0;
        }
    }

    pcl::copyPointCloud(*pointCloud, *colorCloud);
    pcl::copyPointCloud(*pointCloud, *scene);
    // get minimum and maximum
    PointC min_pt, max_pt;
    pcl::getMinMax3D(*colorCloud, min_pt, max_pt);
    //    float minMax = max_pt.z - min_pt.z;
    //    std::cout<<"max_pt.z: "<<max_pt.z<<", min_pt.z"<<min_pt.z;
    float minMax = max_pt.z + 2.0;
    for(int i=0; i<colorCloud->points.size();i++)
    {
        PointC *colorPt = &colorCloud->points.at(i);
        float rgbValue[3] = {.0, .0, .0};
        float colorValue = (colorPt->z - min_pt.z)/minMax;
        getHeatMapColor(colorValue, rgbValue[0], rgbValue[1], rgbValue[2]);
        colorPt->r = int(rgbValue[0]);
        colorPt->g = int(rgbValue[1]);
        colorPt->b = int(rgbValue[2]);
    }
    //    showPointCloud();
    pclVisualizer->removePointCloud("newPc");
    //    KittiPointCloudColorHandlerCustom colorHandler(pointCloud, 128, 128, 128);
    //    pclVisualizer->addPointCloud<KittiPoint>(pointCloud, colorHandler, "newPc");
    pclVisualizer->addPointCloud(colorCloud, "newPc");
    //    pclVisualizer->updatePointCloud<KittiPoint>(pointCloud, colorHandler,"newPc");
    ui->qvtkWidget_pclViewer->update();
}

PointCloudC::Ptr
KittiVisualizerQt::displayPointClouds(PointCloudC::Ptr inputClouds, std::string cloudName,
                                      bool heightColor)
{
    PointCloudC::Ptr inputCloudsColor
            (new PointCloudC);
    pcl::copyPointCloud(*inputClouds, *inputCloudsColor);

    //    std::cout<<"heightColor: "<<heightColor<<std::endl;
    if(heightColor)
    {
        //        std::cout<<"display with height color ...\n";
        for(int i=0; i<inputCloudsColor->points.size(); i++)
        {
            PointC *point = &inputCloudsColor->points.at(i);
            if( (*point).z <-2.5)
            {
                (*point).x = 0; (*point).y = 0; (*point).z = 0;
            }
        }
        // get minimum and maximum
        PointC min_pt, max_pt;
        pcl::getMinMax3D(*inputCloudsColor, min_pt, max_pt);
        float minMax = max_pt.z + 2.0;
        for(int i=0; i<inputCloudsColor->points.size();i++)
        {
            PointC *colorPt = &inputCloudsColor->points.at(i);
            float rgbValue[3] = {.0, .0, .0};
            float colorValue = (colorPt->z - min_pt.z)/minMax;
            getHeatMapColor(colorValue, rgbValue[0], rgbValue[1], rgbValue[2]);
            colorPt->r = int(rgbValue[0]);
            colorPt->g = int(rgbValue[1]);
            colorPt->b = int(rgbValue[2]);
        }
    }
    //    showPointCloud();
    pclVisualizer->removePointCloud(cloudName.c_str());
    pclVisualizer->addPointCloud(inputCloudsColor, cloudName.c_str());
    pclVisualizer->setPointCloudRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, cloudName.c_str());
    ui->qvtkWidget_pclViewer->update();
    return inputCloudsColor;
}

PointCloudC::Ptr
KittiVisualizerQt::displayPointClouds(PointCloudC::Ptr inputClouds, std::string cloudName, int ptSize)
{
    PointCloudC::Ptr inputCloudsColor
            (new PointCloudC);
    pcl::copyPointCloud(*inputClouds, *inputCloudsColor);

    for(int i=0; i<inputCloudsColor->points.size(); i++)
    {
        PointC *point = &inputCloudsColor->points.at(i);
        if( (*point).z <-2.5)
        {
            (*point).x = 0; (*point).y = 0; (*point).z = 0;
        }
    }
    // get minimum and maximum
    PointC min_pt, max_pt;
    pcl::getMinMax3D(*inputCloudsColor, min_pt, max_pt);
    float minMax = max_pt.z + 2.0;
    for(int i=0; i<inputCloudsColor->points.size();i++)
    {
        PointC *colorPt = &inputCloudsColor->points.at(i);
        float rgbValue[3] = {.0, .0, .0};
        float colorValue = (colorPt->z - min_pt.z)/minMax;
        getHeatMapColor(colorValue, rgbValue[0], rgbValue[1], rgbValue[2]);
        colorPt->r = int(rgbValue[0]);
        colorPt->g = int(rgbValue[1]);
        colorPt->b = int(rgbValue[2]);
    }
    //    showPointCloud();
    pclVisualizer->removePointCloud(cloudName.c_str());
    pclVisualizer->addPointCloud(inputCloudsColor, cloudName.c_str());
    pclVisualizer->setPointCloudRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_POINT_SIZE, ptSize, cloudName.c_str());

    ui->qvtkWidget_pclViewer->update();
    return inputCloudsColor;
}

PointCloudC::Ptr
KittiVisualizerQt::displayPointClouds(PointCloudC::Ptr inputClouds, std::string cloudName,
                                      int ptColor[3])
{
    PointCloudC::Ptr inputCloudsColor
            (new PointCloudC);
    pcl::copyPointCloud(*inputClouds, *inputCloudsColor);

    for(int i=0; i<inputCloudsColor->points.size(); i++)
    {
        PointC *point = &inputCloudsColor->points.at(i);
        if( (*point).z <-2.5)
        {
            (*point).x = 0; (*point).y = 0; (*point).z = 0;
        }
    }
    // get minimum and maximum

    for(int i=0; i<inputCloudsColor->points.size();i++)
    {
        PointC *colorPt = &inputCloudsColor->points.at(i);
        colorPt->r = ptColor[0];
        colorPt->g = ptColor[1];
        colorPt->b = ptColor[2];
    }
    //    showPointCloud();
    pclVisualizer->removePointCloud(cloudName.c_str());
    pclVisualizer->addPointCloud(inputCloudsColor, cloudName.c_str());
    pclVisualizer->setPointCloudRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, cloudName.c_str());

    ui->qvtkWidget_pclViewer->update();
    return inputCloudsColor;
}
void KittiVisualizerQt::on_loadTrkPts_clicked()
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr trkPts(new pcl::PointCloud<pcl::PointXYZ>);
    // load *.pcd file
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"),
                                                    "/home/jiang/CvTools/DenseOpticalFlow/OpticalFlow_CeLiu", tr("Files (*.pcd)"));
    if(fileName.size()<1)
    {
        std::cout<<"trkPts not loaded ...\n";
        return;
    }
    pcl::io::loadPCDFile<pcl::PointXYZ>(fileName.toStdString(), *trkPts);
    //    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> trkcolorHandler(trkPts, 0, 255, 0);
    //    pclVisualizer->addPointCloud<pcl::PointXYZ>(trkPts, trkcolorHandler, "trkPts");
    //    pclVisualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
    //                                                      5, "trkPts");

    colorTrk.reset(new PointCloudC);
    pcl::copyPointCloud(*trkPts, *colorTrk);
    if(colorOpt[0]==0 && colorOpt[1]==0 && colorOpt[2]==0)
    {
        for(int i=0; i<colorTrk->points.size();i++)
        {
            PointC *colorPt = &colorTrk->points.at(i);
            int rgbValue[3] = {0, 0, 0};
            getValueBetweenTwoFixedColors(colorPt->z, rgbValue[0], rgbValue[1], rgbValue[2]);
            colorPt->r = int(rgbValue[0]);
            colorPt->g = int(rgbValue[1]);
            colorPt->b = int(rgbValue[2]);
        }
    }
    else
    {
        for(int i=0; i<colorTrk->points.size();i++)
        {
            PointC *colorPt = &colorTrk->points.at(i);
            colorPt->r = colorOpt[0];
            colorPt->g = colorOpt[1];
            colorPt->b = colorOpt[2];
        }
    }
    pclVisualizer->addPointCloud(colorTrk, "colorTrk");
    pclVisualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                    8, "colorTrk");
    ui->qvtkWidget_pclViewer->update();
}

void KittiVisualizerQt::on_loadTrkPtColor_clicked()
{
    if(colorTrk->points.size()<1)   { return; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr ptsColor(new pcl::PointCloud<pcl::PointXYZ>);
    // load *.pcd file
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"),
                                                    "/home/jiang/CvTools/DenseOpticalFlow/OpticalFlow_CeLiu", tr("Files (*.pcd)"));
    if(fileName.size()<1)
    {
        std::cout<<"trkPts color not loaded ...\n";
        return;
    }
    pcl::io::loadPCDFile<pcl::PointXYZ>(fileName.toStdString(), *ptsColor);
    for(int i=0; i<colorTrk->points.size();i++)
    {
        PointC *colorPt = &colorTrk->points.at(i);
        colorPt->r = int(ptsColor->points.at(i).x);
        colorPt->g = int(ptsColor->points.at(i).y);
        colorPt->b = int(ptsColor->points.at(i).z);
    }
    //    pclVisualizer->addPointCloud(colorTrk, "colorTrk");
    //    pclVisualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
    //                                                    8, "colorTrk");
    pclVisualizer->updatePointCloud(colorTrk, "colorTrk");
    ui->qvtkWidget_pclViewer->update();
}

void KittiVisualizerQt::on_clearTrkPts_clicked()
{
    colorTrk->points.clear();
    pclVisualizer->removePointCloud("colorTrk");
    pclVisualizer->removePointCloud("newTrk");
}

void KittiVisualizerQt::on_addPc_clicked()
{
    //    KittiPointCloud::Ptr cloud(new KittiPointCloud);
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"),
                                                    "/home/jiang/CvDataset/KITTI/tracking_module/training/velodyne", \
                                                    tr("Files (*.bin)"));
    std::fstream file(fileName.toStdString().c_str(), std::ios::in | std::ios::binary);
    if(file.good()){
        pointCloud->clear();
        file.seekg(0, std::ios::beg);
        int i;
        for (i = 0; file.good() && !file.eof(); i++) {
            KittiPoint point;
            file.read((char *) &point.x, 3*sizeof(float));
            file.read((char *) &point.intensity, sizeof(float));
            pointCloud->push_back(point);
        }
        file.close();
    }

    PointCloudC::Ptr colorCloud(new PointCloudC);
    pcl::copyPointCloud(*pointCloud, *colorCloud);
    // get minimum and maximum
    PointC min_pt, max_pt;
    pcl::getMinMax3D(*colorCloud, min_pt, max_pt);
    float minMax = max_pt.z - min_pt.z;
    //    std::cout<<"max_pt.z: "<<max_pt.z<<", min_pt.z"<<min_pt.z;
    for(int i=0; i<colorCloud->points.size();i++)
    {
        PointC *colorPt = &colorCloud->points.at(i);
        float rgbValue[3] = {.0, .0, .0};
        float colorValue = (colorPt->z - min_pt.z)/minMax;
        getHeatMapColor(colorValue, rgbValue[0], rgbValue[1], rgbValue[2]);
        colorPt->r = int(rgbValue[0]);
        colorPt->g = int(rgbValue[1]);
        colorPt->b = int(rgbValue[2]);
    }
    //    showPointCloud();
    //    pclVisualizer->removePointCloud("addedPc");
    //    KittiPointCloudColorHandlerCustom colorHandler(pointCloud, 128, 128, 128);
    //    pclVisualizer->addPointCloud<KittiPoint>(pointCloud, colorHandler, "newPc");
    pclVisualizer->addPointCloud(colorCloud, fileName.right(8).toStdString().c_str());
    //    pclVisualizer->updatePointCloud<KittiPoint>(pointCloud, colorHandler,"newPc");
    ui->qvtkWidget_pclViewer->update();
}

void KittiVisualizerQt::on_deletePc_clicked()
{
    pclVisualizer->removeAllPointClouds();
}

void KittiVisualizerQt::on_colorOpt_activated(const QString &arg1)
{

}

void KittiVisualizerQt::on_colorOpt_currentIndexChanged(int index)
{
    switch(index)
    {
    case 0: {colorOpt[0] = 0;colorOpt[1] = 0;colorOpt[2] = 0;break;}
    case 1: {colorOpt[0] = 255;colorOpt[1] = 0;colorOpt[2] = 0;break;}
    case 2: {colorOpt[0] = 0;colorOpt[1] = 255;colorOpt[2] = 0;break;}
    case 3: {colorOpt[0] = 0;colorOpt[1] = 0;colorOpt[2] = 255;break;}
    }
}

void KittiVisualizerQt::on_addTrkPts_clicked()
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr trkPts(new pcl::PointCloud<pcl::PointXYZ>);
    // load *.pcd file
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"),
                                                    "/home/jiang/CvTools/DenseOpticalFlow/OpticalFlow_CeLiu", tr("Files (*.pcd)"));
    if(fileName.size()<1)
    {
        std::cout<<"trkPts not loaded ...\n";
        return;
    }
    pcl::io::loadPCDFile<pcl::PointXYZ>(fileName.toStdString(), *trkPts);


    PointCloudC::Ptr trkPtsColor(new PointCloudC);
    pcl::copyPointCloud(*trkPts, *trkPtsColor);
    if(colorOpt[0]==0 && colorOpt[1]==0 && colorOpt[2]==0)
    {
        for(int i=0; i<trkPtsColor->points.size();i++)
        {
            PointC *colorPt = &trkPtsColor->points.at(i);
            int rgbValue[3] = {0, 0, 0};
            getValueBetweenTwoFixedColors(colorPt->z, rgbValue[0], rgbValue[1], rgbValue[2]);
            colorPt->r = int(rgbValue[0]);
            colorPt->g = int(rgbValue[1]);
            colorPt->b = int(rgbValue[2]);
        }
    }
    else
    {
        for(int i=0; i<trkPtsColor->points.size();i++)
        {
            PointC *colorPt = &trkPtsColor->points.at(i);
            colorPt->r = colorOpt[0];
            colorPt->g = colorOpt[1];
            colorPt->b = colorOpt[2];
        }
    }
    pclVisualizer->addPointCloud(trkPtsColor, "newTrk");
    pclVisualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                    8, "newTrk");
    ui->qvtkWidget_pclViewer->update();
}
////////////////////////////////////////////////////////////
/// \brief KittiVisualizerQt::on_segMovObj_clicked
/// Functions for moving object segmentation
/// ////////////////////////////////////////////////////////
void KittiVisualizerQt::on_segMovObj_clicked()
{
    PointCloudC::Ptr cloudSeg(new PointCloudC);
    seeds->points.clear(); cloudSeg->points.clear(); clusterIdx.clear();
    if(scene->points.size()<1)
    {
        // load scene
        QString fileName = QFileDialog::getOpenFileName(this, tr("load scene"),
                                                        "/home/jiang/CvTools/DenseOpticalFlow/OpticalFlow_CeLiu", tr("Files (*.pcd)"));
        if(fileName.size()<1)
        {
            std::cout<<"seed not loaded ...\n";
            return;
        }
        pcl::io::loadPCDFile<PointC>(fileName.toStdString(), *scene);
        colorCloud.reset(new PointCloudC);
        colorCloud = displayPointClouds(scene, "newPc", 0);
    }

    if(colorTrk->points.size()>1)
    {
        pcl::copyPointCloud(*colorTrk, *seeds);
    }
    else{
        // load seeds
        QString fileName = QFileDialog::getOpenFileName(this, tr("load motion segmentation seeds"),
                                                        "/home/jiang/CvTools/DenseOpticalFlow/OpticalFlow_CeLiu", tr("Files (*.pcd)"));
        if(fileName.size()<1)
        {
            std::cout<<"seed not loaded ...\n";
            return;
        }
        pcl::io::loadPCDFile<PointC>(fileName.toStdString(), *seeds);
        pcl::copyPointCloud(*seeds,*colorTrk);
    }

    PointCloudProcessing segMot;
    std::cout<<"start region growing...\n";
    segMot.removeBadSeed(seeds, searchRadius);
    segMot.pclRegionGrow(scene, seeds, growSpeed, searchRadius, heightThd, cloudSeg, clusterIdx);
    std::cout<<"cluster size: "<<cloudSeg->points.size();
    pclVisualizer->removePointCloud("segMot");
    pcl::visualization::PointCloudColorHandlerCustom<PointC> cloudSegCH(cloudSeg, 255, 0, 0);
    pclVisualizer->addPointCloud<PointC>(cloudSeg, cloudSegCH, "segMot");
    pclVisualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                    5, "segMot");
    ui->qvtkWidget_pclViewer->update();
}

void KittiVisualizerQt::on_regionGrowR_editingFinished()
{
    searchRadius = ui->regionGrowR->text().toFloat();
    std::cout<<"set searchRadius = "<<searchRadius<<"\n";
}

void KittiVisualizerQt::on_heightThd_editingFinished()
{
    heightThd = ui->heightThd->text().toFloat();
    std::cout<<"set heightThd = "<<heightThd<<"\n";
}

void KittiVisualizerQt::on_growSpeed_editingFinished()
{
    growSpeed = ui->growSpeed->text().toFloat();
    std::cout<<"set growSpeed = "<<growSpeed<<"\n";
}

void KittiVisualizerQt::on_removeMot_clicked()
{
    if(clusterIdx.size()<1) {   return;    }
    std::set<int>::iterator clusterIdxIter = clusterIdx.begin();
    for(int i=0; i<clusterIdx.size(); i++, clusterIdxIter++)
    {
        PointC *pointMot = &colorCloud->points.at(*clusterIdxIter);
        if((*pointMot).z > heightThd)
        {
            (*pointMot).r = 0; (*pointMot).g = 0; (*pointMot).b = 0;
        }
    }
    on_clearTrkPts_clicked();
    pclVisualizer->removePointCloud("segMot");
    pclVisualizer->updatePointCloud(colorCloud,"newPc");
    ui->qvtkWidget_pclViewer->update();
}

void KittiVisualizerQt::on_showMot_clicked()
{
    PointCloudC::Ptr motColor(new PointCloudC);
    if(clusterIdx.size()<1) {   return;    }
    if(colorCloud->size()<1){return;}
    std::set<int>::iterator clusterIdxIter = clusterIdx.begin();
    for(int i=0; i<clusterIdx.size(); i++, clusterIdxIter++)
    {
        PointC pointMot = colorCloud->points.at(*clusterIdxIter);
        if(pointMot.z > heightThd)
        {
            motColor->push_back(pointMot);
        }
    }
    pclVisualizer->removeAllPointClouds();
    std::cout<<"add motColor\n";
    pclVisualizer->addPointCloud(motColor, "motColor");
    ui->qvtkWidget_pclViewer->update();
}


void KittiVisualizerQt::removeMotions(PointCloudC::Ptr &scene, std::set<int> clusterIdx)
{
    std::cout<<"motion size: "<<clusterIdx.size()<<std::endl;
    if(clusterIdx.size()<1) {   return;    }
    std::set<int>::iterator clusterIdxIter = clusterIdx.end();
    while(clusterIdxIter!=clusterIdx.begin())
    {
        --clusterIdxIter;
        PointC *pointMot = &scene->points.at(*clusterIdxIter);
        if((*pointMot).z > heightThd)
        {
            scene->points.erase(scene->points.begin()+*clusterIdxIter);
            scene->width--;
        }
    }
    //    if(clusterIdx.size()<1) {   return;    }
    //    std::set<int>::iterator clusterIdxIter = clusterIdx.begin();
    //    for(int i=0; i<clusterIdx.size(); i++, clusterIdxIter++)
    //    {
    //        PointT *pointMot = &scene->points.at(*clusterIdxIter);
    //        if((*pointMot).z > heightThd)
    //        {
    //            (*pointMot).z += 2.0;
    //        }
    //    }
}


void KittiVisualizerQt::on_frontView_clicked()
{
    for(int i=0; i<colorCloud->points.size(); i++)
    {
        PointC *point = &colorCloud->points.at(i);
        if( (*point).x<=0.1 )
        {
            (*point).r = 0; (*point).g = 0; (*point).b = 0;
        }
    }
    pclVisualizer->updatePointCloud(colorCloud, "newPc");
    ui->qvtkWidget_pclViewer->update();
}

void KittiVisualizerQt::on_FoV3d2d_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"),
                                                    "/home/jiang/CvTools/DenseOpticalFlow/OpticalFlow_CeLiu", \
                                                    tr("Files (*.pcd)"));

    if(fileName.size()<1)
    {
        std::cout<<"FoV3d2d point cloud not loaded ...\n";
        return;
    }
    scene->points.clear(); pointCloud->points.clear();
    pcl::io::loadPCDFile<PointC>(fileName.toStdString(), *scene);
    for(int i=0; i<scene->points.size(); i++)
    {
        PointC *point = &scene->points.at(i);
        if( (*point).z <-2.5)
        {
            (*point).x = 0; (*point).y = 0; (*point).z = 0;
        }
    }
    pcl::copyPointCloud(*scene, *colorCloud);
    pcl::copyPointCloud(*scene, *pointCloud);
    // get minimum and maximum
    PointC min_pt, max_pt;
    pcl::getMinMax3D(*colorCloud, min_pt, max_pt);
    float minMax = max_pt.z + 2.0;
    for(int i=0; i<colorCloud->points.size();i++)
    {
        PointC *colorPt = &colorCloud->points.at(i);
        float rgbValue[3] = {.0, .0, .0};
        float colorValue = (colorPt->z - min_pt.z)/minMax;
        getHeatMapColor(colorValue, rgbValue[0], rgbValue[1], rgbValue[2]);
        colorPt->r = int(rgbValue[0]);
        colorPt->g = int(rgbValue[1]);
        colorPt->b = int(rgbValue[2]);
    }
    pclVisualizer->removePointCloud("newPc");
    pclVisualizer->addPointCloud(colorCloud, "newPc");
    ui->qvtkWidget_pclViewer->update();
}

void KittiVisualizerQt::on_addFoV3d2d_clicked()
{
    PointCloudC::Ptr addFoV3d2d(new PointCloudC);
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"),
                                                    "/home/jiang/CvTools/DenseOpticalFlow/OpticalFlow_CeLiu", \
                                                    tr("Files (*.pcd)"));

    if(fileName.size()<1)
    {
        std::cout<<"FoV3d2d point cloud not added ...\n";
        return;
    }
    addFoV3d2d->points.clear();
    pcl::io::loadPCDFile<PointC>(fileName.toStdString(), *addFoV3d2d);
    for(int i=0; i<addFoV3d2d->points.size(); i++)
    {
        PointC *point = &addFoV3d2d->points.at(i);
        if( (*point).z <-2.5)
        {
            (*point).x = 0; (*point).y = 0; (*point).z = 0;
        }
    }

    // get minimum and maximum
    PointC min_pt, max_pt;
    pcl::getMinMax3D(*addFoV3d2d, min_pt, max_pt);
    float minMax = max_pt.z + 2.0;
    for(int i=0; i<addFoV3d2d->points.size();i++)
    {
        PointC *colorPt = &addFoV3d2d->points.at(i);
        float rgbValue[3] = {.0, .0, .0};
        float colorValue = (colorPt->z - min_pt.z)/minMax;
        getHeatMapColor(colorValue, rgbValue[0], rgbValue[1], rgbValue[2]);
        colorPt->r = int(rgbValue[0]);
        colorPt->g = int(rgbValue[1]);
        colorPt->b = int(rgbValue[2]);
    }
    std::cout<<"addFoV3d2d size: "<< addFoV3d2d->points.size()<<std::endl;
    pclVisualizer->addPointCloud(addFoV3d2d, fileName.toStdString().c_str());
    std::cout<<"loaded: "<<fileName.toStdString().c_str()<<"\n";
    ui->qvtkWidget_pclViewer->update();
}
bool fexists(const std::string& filename) {
    std::ifstream ifile(filename.c_str());
    return ifile;
}
PointCloudC::Ptr
KittiVisualizerQt::loadPointClouds(std::string &filesName)
{
    PointCloudC::Ptr addFoV3d2d(new PointCloudC);
    QString fileName;
    if(!fexists(filesName.c_str()))
    {
        fileName = QFileDialog::getOpenFileName(this, tr(filesName.c_str()),
                                                "/home/jiang/CvTools/DenseOpticalFlow/OpticalFlow_CeLiu/MSresult/people1car1_1_10/", \
                                                tr("Files (*.pcd)"));

        if(fileName.size()<1)
        {
            std::cout<<"PointClouds not added ...\n";
            return addFoV3d2d;
        }
    }else{fileName = QString(filesName.c_str());}

    addFoV3d2d->points.clear();
    pcl::io::loadPCDFile<PointC>(fileName.toStdString(), *addFoV3d2d);
    int fileNamelen = fileName.toStdString().length();
    std::string textureName = fileName.toStdString().substr(0, fileNamelen-4) + "_texture.pcd";
    std::cout<<"textureName: "<<textureName<<"\n";
    if(!fexists(textureName.c_str()))
    {
        std::cout<<"Scene texture not exist, mapping color with height ...\n";
        for(int i=0; i<addFoV3d2d->points.size(); i++)
        {
            PointC *point = &addFoV3d2d->points.at(i);
            if( (*point).z <-2.5)
            {
                (*point).x = 0; (*point).y = 0; (*point).z = 0;
            }
        }

        // get minimum and maximum
        PointC min_pt, max_pt;
        pcl::getMinMax3D(*addFoV3d2d, min_pt, max_pt);
        float minMax = max_pt.z + 2.0;
        for(int i=0; i<addFoV3d2d->points.size();i++)
        {
            PointC *colorPt = &addFoV3d2d->points.at(i);
            float rgbValue[3] = {.0, .0, .0};
            float colorValue = (colorPt->z - min_pt.z)/minMax;
            getHeatMapColor(colorValue, rgbValue[0], rgbValue[1], rgbValue[2]);
            colorPt->r = int(rgbValue[0]);
            colorPt->g = int(rgbValue[1]);
            colorPt->b = int(rgbValue[2]);
        }
    }else
    {
        std::cout<<"Scene texture exists, mapping texture ...\n";
        PointCloudT::Ptr cloudTexture(new PointCloudT);
        pcl::io::loadPCDFile<PointT>(textureName, *cloudTexture);
        for(int i=0; i<addFoV3d2d->points.size();i++)
        {
            PointC *colorPt = &addFoV3d2d->points.at(i);
            PointT ptTexture = cloudTexture->points.at(i);
            colorPt->r = int(ptTexture.x);
            colorPt->g = int(ptTexture.y);
            colorPt->b = int(ptTexture.z);
            //            std::cout<<"rgb: "<<int(colorPt->r)<<" "<<colorPt->g<<" "<<colorPt->b<<" ";
        }
    }
    std::cout<<"addFoV3d2d size: "<< addFoV3d2d->points.size()<<std::endl;
    pclVisualizer->addPointCloud(addFoV3d2d, fileName.toStdString().c_str());
    std::cout<<"loaded: "<<fileName.toStdString().c_str()<<"\n";
    ui->qvtkWidget_pclViewer->update();

    filesName.clear(); filesName = fileName.toStdString();
    return addFoV3d2d;
}

void KittiVisualizerQt::on_load2scenes_clicked()
{
    // load 2 scenes
    str2SceneRansacParams->sceneRef = loadPointClouds(str2SceneRansacParams->sceneRefName);
    str2SceneRansacParams->sceneNew = loadPointClouds(str2SceneRansacParams->sceneNewName);

    // load 2 scenes' correspondences
    str2SceneRansacParams->corrRef = loadPointClouds(str2SceneRansacParams->corrRefName);
    str2SceneRansacParams->corrNew = loadPointClouds(str2SceneRansacParams->corrNewName);

    str2SceneRansacParams->motSeedsRef = loadPointClouds(str2SceneRansacParams->motSeedsRefName);
    str2SceneRansacParams->motSeedsNew = loadPointClouds(str2SceneRansacParams->motSeedsNewName);

    pcl::copyPointCloud(*(str2SceneRansacParams->sceneRef),
                        *(str2SceneRansacParams->sceneNoMotRef));

    pcl::copyPointCloud(*(str2SceneRansacParams->sceneNew),
                        *(str2SceneRansacParams->sceneNoMotNew));

    //    pclVisualizer->removePointCloud(str2SceneRansacParams->sceneNewName.c_str());
    //    displayPointClouds(str2SceneRansacParams->motSeedsRef, str2SceneRansacParams->motSeedsRefName, 5);

    // motion segmention
    PointCloudProcessing registrationObj;
    registrationObj.removeBadSeed(str2SceneRansacParams->motSeedsRef, searchRadius);
    registrationObj.pclRegionGrow(str2SceneRansacParams->sceneRef,
                                  str2SceneRansacParams->motSeedsRef,
                                  growSpeed, searchRadius, heightThd, \
                                  str2SceneRansacParams->sceneMotRef,
                                  str2SceneRansacParams->motIdxRef);
    //    std::cout<<"size of ref scene: "<<str2SceneRansacParams->sceneRef->points.size()<<"\n";
    removeMotions(str2SceneRansacParams->sceneNoMotRef, str2SceneRansacParams->motIdxRef);
    //    std::cout<<"size of ref scene no motion: "<<str2SceneRansacParams->sceneNoMotRef->points.size()<<"\n";

    registrationObj.removeBadSeed(str2SceneRansacParams->motSeedsNew, searchRadius);
    registrationObj.pclRegionGrow(str2SceneRansacParams->sceneNew,
                                  str2SceneRansacParams->motSeedsNew,
                                  growSpeed, searchRadius, heightThd, \
                                  str2SceneRansacParams->sceneMotNew,
                                  str2SceneRansacParams->motIdxNew);

    //    std::cout<<"size of new scene: "<<str2SceneRansacParams->sceneNew->points.size()<<"\n";
    removeMotions(str2SceneRansacParams->sceneNoMotNew, str2SceneRansacParams->motIdxNew);
    //    std::cout<<"size of new scene no motion: "<<str2SceneRansacParams->sceneNoMotNew->points.size()<<"\n";

    pclVisualizer->removePointCloud(str2SceneRansacParams->sceneRefName.c_str());
    pclVisualizer->removePointCloud(str2SceneRansacParams->corrRefName.c_str());
    pclVisualizer->removePointCloud(str2SceneRansacParams->motSeedsRefName.c_str());
    pclVisualizer->removePointCloud(str2SceneRansacParams->sceneNewName.c_str());
    pclVisualizer->removePointCloud(str2SceneRansacParams->corrNewName.c_str());
    pclVisualizer->removePointCloud(str2SceneRansacParams->motSeedsNewName.c_str());

    registrationObj.normalizePointClouds(str2SceneRansacParams->corrRef,
                                         str2SceneRansacParams->normalizationMat);

    std::cout<<"normalization transmat: "<<str2SceneRansacParams->normalizationMat<<"\n";

    pcl::transformPointCloud(*str2SceneRansacParams->corrNew, *str2SceneRansacParams->corrNew,
                             str2SceneRansacParams->normalizationMat);
    pcl::transformPointCloud(*str2SceneRansacParams->sceneNoMotRef, *str2SceneRansacParams->sceneNoMotRef,
                             str2SceneRansacParams->normalizationMat);
    pcl::transformPointCloud(*str2SceneRansacParams->sceneNoMotNew, *str2SceneRansacParams->sceneNoMotNew,
                             str2SceneRansacParams->normalizationMat);

    displayPointClouds(str2SceneRansacParams->sceneNoMotRef,
                       str2SceneRansacParams->sceneRefName, false);
    displayPointClouds(str2SceneRansacParams->sceneNoMotNew,
                       str2SceneRansacParams->sceneNewName, false);

    displayPointClouds(str2SceneRansacParams->corrRef, str2SceneRansacParams->corrRefName, 10);
    displayPointClouds(str2SceneRansacParams->corrNew, str2SceneRansacParams->corrNewName, 10);
}

void KittiVisualizerQt::on_register2scenes_clicked()
{
    PointCloudProcessing registrationObj;
    // register 2 scenes
    std::cout<<"start 2 scenes registration...\n";
    registrationObj.register2ScenesRansac(str2SceneRansacParams->sceneNoMotRef,
                                          str2SceneRansacParams->sceneNoMotNew,
                                          str2SceneRansacParams->corrRef,
                                          str2SceneRansacParams->corrNew,
                                          str2SceneRansacParams->inlrThdRansac,
                                          str2SceneRansacParams->sampleNbRansac,
                                          str2SceneRansacParams->inlrRateRansac,
                                          str2SceneRansacParams->maxIterRansac,
                                          str2SceneRansacParams->transMat,
                                          str2SceneRansacParams->registeredScene);

    std::cout<<"2 scenes registration finished...\n";
    // remove preprocessing data visualization
    pclVisualizer->removePointCloud(str2SceneRansacParams->sceneRefName.c_str());
    pclVisualizer->removePointCloud(str2SceneRansacParams->sceneNewName.c_str());
    pclVisualizer->removePointCloud(str2SceneRansacParams->corrRefName.c_str());
    pclVisualizer->removePointCloud(str2SceneRansacParams->corrNewName.c_str());
    //    pclVisualizer->removePointCloud(str2SceneRansacParams->motSeedsRefName.c_str());
    //    pclVisualizer->removePointCloud(str2SceneRansacParams->motSeedsNewName.c_str());
    pclVisualizer->removePointCloud(str2SceneRansacParams->registeredName.c_str());
    //    pclVisualizer->addPointCloud(str2SceneRansacParams->registeredScene,
    //                                 str2SceneRansacParams->registeredName.c_str());
    displayPointClouds(str2SceneRansacParams->registeredScene,
                       str2SceneRansacParams->registeredName.c_str(), false);
    pcl::transformPointCloud(*str2SceneRansacParams->corrNew, *str2SceneRansacParams->corrNew,
                             str2SceneRansacParams->transMat);
    int ptColorRef[3] = {255, 0, 0};
    int ptColorNew[3] = {0, 255, 0};
    displayPointClouds(str2SceneRansacParams->corrRef, str2SceneRansacParams->corrRefName, ptColorRef);
    displayPointClouds(str2SceneRansacParams->corrNew, str2SceneRansacParams->corrNewName, ptColorNew);
    ui->qvtkWidget_pclViewer->update();
}

void KittiVisualizerQt::on_inlrRateRansac_editingFinished()
{
    str2SceneRansacParams->inlrRateRansac = ui->inlrRateRansac->text().toFloat();
}

void KittiVisualizerQt::on_smpRateRansac_editingFinished()
{
    str2SceneRansacParams->smpRateRansac = ui->smpRateRansac->text().toFloat();
}

void KittiVisualizerQt::on_inlrThdRansac_editingFinished()
{
    str2SceneRansacParams->inlrThdRansac = ui->inlrThdRansac->text().toFloat();
}

void KittiVisualizerQt::on_maxIterRansac_editingFinished()
{
    str2SceneRansacParams->maxIterRansac = ui->maxIterRansac->text().toInt();
}


void KittiVisualizerQt::on_loadRefScene_clicked()
{
    // load reference scene for registration
    str2SceneRansacParams->sceneRef = loadPointClouds(str2SceneRansacParams->sceneRefName);

    int fileNamelen = str2SceneRansacParams->sceneRefName.length();
    std::string basedDir = str2SceneRansacParams->sceneRefName.substr(0, fileNamelen-4);
    str2SceneRansacParams->corrRefName = basedDir + "_bkg.pcd";
    str2SceneRansacParams->motSeedsRefName = basedDir + "_mot.pcd";

    str2SceneRansacParams->corrRef  = loadPointClouds(str2SceneRansacParams->corrRefName);
    str2SceneRansacParams->motSeedsRef = loadPointClouds(str2SceneRansacParams->motSeedsRefName);
    pcl::copyPointCloud(*(str2SceneRansacParams->sceneRef),
                        *(str2SceneRansacParams->sceneNoMotRef));

    // motion segmention
    PointCloudProcessing registrationObj;
    registrationObj.removeBadSeed(str2SceneRansacParams->motSeedsRef,searchRadius);
    registrationObj.pclRegionGrow(str2SceneRansacParams->sceneRef,
                                  str2SceneRansacParams->motSeedsRef,
                                  growSpeed, searchRadius, heightThd, \
                                  str2SceneRansacParams->sceneMotRef,
                                  str2SceneRansacParams->motIdxRef);
    removeMotions(str2SceneRansacParams->sceneNoMotRef, str2SceneRansacParams->motIdxRef);
    pclVisualizer->removePointCloud(str2SceneRansacParams->sceneRefName.c_str());
    pclVisualizer->removePointCloud(str2SceneRansacParams->motSeedsRefName.c_str());
    pclVisualizer->removePointCloud(str2SceneRansacParams->corrRefName.c_str());

    registrationObj.normalizePointClouds(str2SceneRansacParams->corrRef,
                                         str2SceneRansacParams->normalizationMat);
    std::cout<<"normalization transmat: "<<str2SceneRansacParams->normalizationMat<<"\n";
    pcl::transformPointCloud(*str2SceneRansacParams->sceneNoMotRef, *str2SceneRansacParams->sceneNoMotRef,
                             str2SceneRansacParams->normalizationMat);

    displayPointClouds(str2SceneRansacParams->sceneNoMotRef,
                       str2SceneRansacParams->sceneRefName, false);

    displayPointClouds(str2SceneRansacParams->corrRef, str2SceneRansacParams->corrRefName, 10);

}

void KittiVisualizerQt::on_addRegistScene_clicked()
{
    str2SceneRansacParams->sceneNewName     = "load New Scene";
    str2SceneRansacParams->corrNewName      = "load New Correspondences";
    str2SceneRansacParams->motSeedsNewName  = "load New Motion Seeds";
    str2SceneRansacParams->sceneNew = loadPointClouds(str2SceneRansacParams->sceneNewName);
    std::cout<<"sceneNew size: "<<str2SceneRansacParams->sceneNew->points.size()<<"\n";
    int fileNamelen = str2SceneRansacParams->sceneNewName.length();
    std::string basedDir = str2SceneRansacParams->sceneNewName.substr(0, fileNamelen-4);
    str2SceneRansacParams->corrNewName = basedDir + "_bkg.pcd";
    str2SceneRansacParams->motSeedsNewName = basedDir + "_mot.pcd";

    str2SceneRansacParams->corrNew = loadPointClouds(str2SceneRansacParams->corrNewName);
    str2SceneRansacParams->motSeedsNew = loadPointClouds(str2SceneRansacParams->motSeedsNewName);
    pcl::copyPointCloud(*(str2SceneRansacParams->sceneNew),
                        *(str2SceneRansacParams->sceneNoMotNew));
    std::cout<<"loading new registration scene done...\n";
    PointCloudProcessing registrationObj;
    registrationObj.removeBadSeed(str2SceneRansacParams->motSeedsNew,searchRadius);
    registrationObj.pclRegionGrow(str2SceneRansacParams->sceneNew,
                                  str2SceneRansacParams->motSeedsNew,
                                  growSpeed, searchRadius, heightThd, \
                                  str2SceneRansacParams->sceneMotNew,
                                  str2SceneRansacParams->motIdxNew);
    std::cout<<"region growing new registration scene done...\n";
    removeMotions(str2SceneRansacParams->sceneNoMotNew, str2SceneRansacParams->motIdxNew);
    pclVisualizer->removePointCloud(str2SceneRansacParams->sceneNewName.c_str());
    pclVisualizer->removePointCloud(str2SceneRansacParams->corrNewName.c_str());
    pclVisualizer->removePointCloud(str2SceneRansacParams->motSeedsNewName.c_str());
    std::cout<<"removing motions in new registration scene done...\n";

    pcl::transformPointCloud(*str2SceneRansacParams->corrNew, *str2SceneRansacParams->corrNew,
                             str2SceneRansacParams->normalizationMat);
    pcl::transformPointCloud(*str2SceneRansacParams->sceneNoMotNew, *str2SceneRansacParams->sceneNoMotNew,
                             str2SceneRansacParams->normalizationMat);
    std::cout<<"normalizing new registration scene done...\n";

    displayPointClouds(str2SceneRansacParams->sceneNoMotNew,
                       str2SceneRansacParams->sceneNewName, false);

    int corrNewColor[3] = {255, 0, 0};
    displayPointClouds(str2SceneRansacParams->corrNew, str2SceneRansacParams->corrNewName, corrNewColor);
}

void KittiVisualizerQt::on_nScenesRansac_clicked()
{
    PointCloudProcessing registrationObj;
    // register 2 scenes
    std::cout<<"start 2 scenes registration...\n";
    registrationObj.register2ScenesRansac(str2SceneRansacParams->sceneNoMotRef,
                                          str2SceneRansacParams->sceneNoMotNew,
                                          str2SceneRansacParams->corrRef,
                                          str2SceneRansacParams->corrNew,
                                          str2SceneRansacParams->inlrThdRansac,
                                          str2SceneRansacParams->sampleNbRansac,
                                          str2SceneRansacParams->inlrRateRansac,
                                          str2SceneRansacParams->maxIterRansac,
                                          str2SceneRansacParams->transMat,
                                          str2SceneRansacParams->registeredScene);
    std::cout<<"2 scenes registration finished...\n";
    // remove preprocessing data visualization
    pclVisualizer->removePointCloud(str2SceneRansacParams->sceneRefName.c_str());
    pclVisualizer->removePointCloud(str2SceneRansacParams->sceneNewName.c_str());
    pclVisualizer->removePointCloud(str2SceneRansacParams->corrRefName.c_str());
    pclVisualizer->removePointCloud(str2SceneRansacParams->corrNewName.c_str());
    pclVisualizer->removePointCloud(str2SceneRansacParams->motSeedsRefName.c_str());
    pclVisualizer->removePointCloud(str2SceneRansacParams->motSeedsNewName.c_str());
    pclVisualizer->removePointCloud(str2SceneRansacParams->registeredName.c_str());

    displayPointClouds(str2SceneRansacParams->registeredScene,
                       str2SceneRansacParams->registeredName.c_str(), false);
    pcl::transformPointCloud(*str2SceneRansacParams->corrNew, *str2SceneRansacParams->corrNew,
                             str2SceneRansacParams->transMat);
    int ptColorRef[3] = {255, 0, 0};
    int ptColorNew[3] = {0, 255, 0};
    //    displayPointClouds(str2SceneRansacParams->corrRef, str2SceneRansacParams->corrRefName, ptColorRef);
    //    displayPointClouds(str2SceneRansacParams->corrNew, str2SceneRansacParams->corrNewName, ptColorNew);
    ui->qvtkWidget_pclViewer->update();
}

void KittiVisualizerQt::on_saveRegist_clicked()
{
    int refNameLen = str2SceneRansacParams->sceneRefName.length();
    std::string baseName = str2SceneRansacParams->sceneNewName.substr(0, refNameLen-4);
    char buffer[256]; sprintf(buffer, "%02d", str2SceneRansacParams->registLength + 1);
    std::string newIdx(buffer);
    std::string saveNameRegistScene = baseName + "_len" + newIdx + ".pcd";
    std::string saveNameTransMat = baseName + "_len" + newIdx + "_transMat.txt";
    std::string saveNameNrmlMat  = baseName + "_len" + newIdx + "_NrmlMat.txt";
    voxelDensityFiltering(str2SceneRansacParams->registeredScene);
    pcl::io::savePCDFileASCII(saveNameRegistScene.c_str(),
                              *str2SceneRansacParams->registeredScene);

    // save normalization matrix
    std::ofstream saveRefNrmlMat;
    saveRefNrmlMat.open(saveNameNrmlMat.c_str());
    for(int i=0; i<4; i++)
    {
        for(int j=0; j<4; j++)
        {
            saveRefNrmlMat << str2SceneRansacParams->normalizationMat(i,j)<<" ";
        }
        saveRefNrmlMat << "\n";
    }
    saveRefNrmlMat.close();

    // save last transformation matrix
    std::ofstream saveLastTransMat;
    saveLastTransMat.open(saveNameTransMat.c_str());
    for(int i=0; i<4; i++)
    {
        for(int j=0; j<4; j++)
        {
            saveLastTransMat << str2SceneRansacParams->transMat(i,j)<<" ";
        }
        saveLastTransMat << "\n";
    }
    saveLastTransMat.close();
}

void KittiVisualizerQt::voxelDensityFiltering(PointCloudC::Ptr& registeredScene)
{
    pcl::PCLPointCloud2::Ptr filteredRegist2 (new pcl::PCLPointCloud2);
    // convert pointcloud to pointcloud2
    pcl::toPCLPointCloud2(*registeredScene, *filteredRegist2);

    // Create the filtering object
    pcl::VoxelGrid<pcl::PCLPointCloud2> voxGrid;
    voxGrid.setInputCloud (filteredRegist2);
    voxGrid.setLeafSize (0.02, 0.02, 0.02);
    voxGrid.filter (*filteredRegist2);

    std::cout<<"Registered cloud size before voxel filtering: "<<
               registeredScene->points.size()<<"\n";
    // convert pointcloud2 to pointcloud
    registeredScene->points.clear();
    pcl::fromPCLPointCloud2(*filteredRegist2, *registeredScene);

    std::cout<<"Registered cloud size after voxel filtering: "<<registeredScene->points.size()<<"\n";
}
///////////////////////////////////////////////////////////////////////////////
/// \brief KittiVisualizerQt::on_seqRegist_clicked
///
void KittiVisualizerQt::on_seqRegist_clicked()
{
    str2SceneRansacParams->sceneRefName = "load Reference Frame ...\n";
    str2SceneRansacParams->registeredScene->points.clear();
    // load reference scene for registration
    str2SceneRansacParams->sceneRef = loadPointClouds(str2SceneRansacParams->sceneRefName);
    str2SceneRansacParams->registLength = QInputDialog::getInt(0, "PointCloud Registration",
                                                               "length of sequence:", 1);
    std::cout<<"length of registration sequence: "<<str2SceneRansacParams->registLength<<"\n";
    int fileNamelen = str2SceneRansacParams->sceneRefName.length();
    std::string basedDir = str2SceneRansacParams->sceneRefName.substr(0, fileNamelen-4);
    str2SceneRansacParams->corrRefName = basedDir + "_bkg.pcd";
    str2SceneRansacParams->motSeedsRefName = basedDir + "_mot.pcd";

    str2SceneRansacParams->corrRef  = loadPointClouds(str2SceneRansacParams->corrRefName);
    str2SceneRansacParams->motSeedsRef = loadPointClouds(str2SceneRansacParams->motSeedsRefName);
    pcl::copyPointCloud(*(str2SceneRansacParams->sceneRef),
                        *(str2SceneRansacParams->sceneNoMotRef));

    // motion segmention
    PointCloudProcessing registrationObj;
    registrationObj.removeBadSeed(str2SceneRansacParams->motSeedsRef,searchRadius);
    registrationObj.pclRegionGrow(str2SceneRansacParams->sceneRef,
                                  str2SceneRansacParams->motSeedsRef,
                                  growSpeed, searchRadius, heightThd, \
                                  str2SceneRansacParams->sceneMotRef,
                                  str2SceneRansacParams->motIdxRef);
    removeMotions(str2SceneRansacParams->sceneNoMotRef, str2SceneRansacParams->motIdxRef);
    pclVisualizer->removePointCloud(str2SceneRansacParams->sceneRefName.c_str());
    pclVisualizer->removePointCloud(str2SceneRansacParams->motSeedsRefName.c_str());
    pclVisualizer->removePointCloud(str2SceneRansacParams->corrRefName.c_str());

    registrationObj.normalizePointClouds(str2SceneRansacParams->corrRef,
                                         str2SceneRansacParams->normalizationMat);
    //    std::cout<<"normalization transmat: "<<str2SceneRansacParams->normalizationMat<<"\n";
    pcl::transformPointCloud(*str2SceneRansacParams->sceneNoMotRef, *str2SceneRansacParams->sceneNoMotRef,
                             str2SceneRansacParams->normalizationMat);

    //    while(loopCondition)
    for(int i=0; i<str2SceneRansacParams->registLength; i++)
    {
        std::string refIdx= str2SceneRansacParams->sceneRefName.substr(fileNamelen-6, fileNamelen-5);
        basedDir = str2SceneRansacParams->sceneRefName.substr(0, fileNamelen-6);
        int rIdx = std::atoi(refIdx.c_str()); ++rIdx;
        char buffer[256]; sprintf(buffer, "%02d", rIdx);
        std::string newIdx(buffer);
        str2SceneRansacParams->sceneNewName    = basedDir + newIdx + ".pcd";
        str2SceneRansacParams->corrNewName     = basedDir + newIdx + "_bkg.pcd";
        str2SceneRansacParams->motSeedsNewName = basedDir + newIdx + "_mot.pcd";
        str2SceneRansacParams->sceneNew = loadPointClouds(str2SceneRansacParams->sceneNewName);
        std::cout<<"sceneNew size: "<<str2SceneRansacParams->sceneNew->points.size()<<"\n";

        str2SceneRansacParams->corrNew = loadPointClouds(str2SceneRansacParams->corrNewName);
        str2SceneRansacParams->motSeedsNew = loadPointClouds(str2SceneRansacParams->motSeedsNewName);
        pcl::copyPointCloud(*(str2SceneRansacParams->sceneNew),
                            *(str2SceneRansacParams->sceneNoMotNew));
        registrationObj.removeBadSeed(str2SceneRansacParams->motSeedsNew,searchRadius);
        registrationObj.pclRegionGrow(str2SceneRansacParams->sceneNew,
                                      str2SceneRansacParams->motSeedsNew,
                                      growSpeed, searchRadius, heightThd, \
                                      str2SceneRansacParams->sceneMotNew,
                                      str2SceneRansacParams->motIdxNew);

        removeMotions(str2SceneRansacParams->sceneNoMotNew, str2SceneRansacParams->motIdxNew);
        pclVisualizer->removePointCloud(str2SceneRansacParams->sceneNewName.c_str());
        pclVisualizer->removePointCloud(str2SceneRansacParams->corrNewName.c_str());
        pclVisualizer->removePointCloud(str2SceneRansacParams->motSeedsNewName.c_str());

        pcl::transformPointCloud(*str2SceneRansacParams->sceneNew, *str2SceneRansacParams->sceneNew,
                                 str2SceneRansacParams->normalizationMat);
        pcl::transformPointCloud(*str2SceneRansacParams->corrNew, *str2SceneRansacParams->corrNew,
                                 str2SceneRansacParams->normalizationMat);
        pcl::transformPointCloud(*str2SceneRansacParams->motSeedsNew, *str2SceneRansacParams->motSeedsNew,
                                 str2SceneRansacParams->normalizationMat);
        pcl::transformPointCloud(*str2SceneRansacParams->sceneNoMotNew, *str2SceneRansacParams->sceneNoMotNew,
                                 str2SceneRansacParams->normalizationMat);

        // register 2 scenes
        registrationObj.register2ScenesRansac(str2SceneRansacParams->sceneNoMotRef,
                                              str2SceneRansacParams->sceneNoMotNew,
                                              str2SceneRansacParams->corrRef,
                                              str2SceneRansacParams->corrNew,
                                              str2SceneRansacParams->inlrThdRansac,
                                              str2SceneRansacParams->sampleNbRansac,
                                              str2SceneRansacParams->inlrRateRansac,
                                              str2SceneRansacParams->maxIterRansac,
                                              str2SceneRansacParams->transMat,
                                              str2SceneRansacParams->registeredScene);

        // remove preprocessing data visualization
        pclVisualizer->removePointCloud(str2SceneRansacParams->sceneRefName.c_str());
        pclVisualizer->removePointCloud(str2SceneRansacParams->sceneNewName.c_str());
        pclVisualizer->removePointCloud(str2SceneRansacParams->corrRefName.c_str());
        pclVisualizer->removePointCloud(str2SceneRansacParams->corrNewName.c_str());
        pclVisualizer->removePointCloud(str2SceneRansacParams->motSeedsRefName.c_str());
        pclVisualizer->removePointCloud(str2SceneRansacParams->motSeedsNewName.c_str());
        pclVisualizer->removePointCloud(str2SceneRansacParams->registeredName.c_str());

        displayPointClouds(str2SceneRansacParams->registeredScene,
                           str2SceneRansacParams->registeredName.c_str(), false);

        // update reference point cloud
        str2SceneRansacParams->sceneRefName    = str2SceneRansacParams->sceneNewName;
        str2SceneRansacParams->corrRefName     = str2SceneRansacParams->corrNewName;
        str2SceneRansacParams->motSeedsRefName = str2SceneRansacParams->motSeedsNewName;

        pcl::transformPointCloud(*str2SceneRansacParams->sceneNew, *str2SceneRansacParams->sceneNew,
                                 str2SceneRansacParams->transMat);
        pcl::transformPointCloud(*str2SceneRansacParams->corrNew, *str2SceneRansacParams->corrNew,
                                 str2SceneRansacParams->transMat);
        pcl::transformPointCloud(*str2SceneRansacParams->motSeedsNew, *str2SceneRansacParams->motSeedsNew,
                                 str2SceneRansacParams->transMat);
        pcl::transformPointCloud(*str2SceneRansacParams->sceneNoMotNew, *str2SceneRansacParams->sceneNoMotNew,
                                 str2SceneRansacParams->transMat);

        pcl::copyPointCloud(*str2SceneRansacParams->sceneNew,
                            *str2SceneRansacParams->sceneRef);
        pcl::copyPointCloud(*str2SceneRansacParams->corrNew,
                            *str2SceneRansacParams->corrRef);
        pcl::copyPointCloud(*str2SceneRansacParams->motSeedsNew,
                            *str2SceneRansacParams->motSeedsRef);
        pcl::copyPointCloud(*str2SceneRansacParams->sceneMotNew,
                            *str2SceneRansacParams->sceneMotRef);
        pcl::copyPointCloud(*str2SceneRansacParams->sceneNoMotNew,
                            *str2SceneRansacParams->sceneNoMotRef);

        std::cout<<"transMat: "<<str2SceneRansacParams->transMat<<"\n";
        //        voxelDensityFiltering(str2SceneRansacParams->registeredScene);
    }
    on_saveRegist_clicked();
}

void KittiVisualizerQt::on_mergeSubSeq_clicked()
{
    PointCloudC::Ptr subSeq1(new PointCloudC), subSeq2(new PointCloudC);
    QString fileName;
    fileName = QFileDialog::getOpenFileName(this, tr("reference scene to merge..."),
                                            "/home/jiang/CvTools/DenseOpticalFlow/OpticalFlow_CeLiu/MSresult/", \
                                            tr("Files (*.pcd)"));

    if(fileName.size()<1) { std::cout<<"PointClouds not added ...\n";  return;  }
    pcl::io::loadPCDFile<PointC>(fileName.toStdString(), *subSeq1);

    // load last frame transMat
    std::string transMatName = fileName.toStdString().substr(0, fileName.length() - 4);
    transMatName = transMatName + "_transMat.txt";

    if(!fexists(transMatName.c_str()))
    {
        QString transfileName;
        transfileName = QFileDialog::getOpenFileName(this, tr("load last frame transMat ..."),
                                                "/home/jiang/CvTools/DenseOpticalFlow/OpticalFlow_CeLiu/MSresult/", \
                                                tr("Files (*.txt)"));
        transMatName = transfileName.toStdString();
    }

    std::ifstream transMatFile(transMatName.c_str());
    Eigen::Matrix4f transMat; transMat.setZero();
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            transMatFile >> transMat(i, j);
        }
    }
    std::cout<<"loaded transMat: "<<transMat<<"\n";

    // load ref frame normalization mat
    std::string refNrmlName = fileName.toStdString().substr(0, fileName.length() - 4);
    refNrmlName = refNrmlName + "_NrmlMat.txt";
    if(!fexists(refNrmlName.c_str()))
    {
        QString refNrmlfileName;
        refNrmlfileName = QFileDialog::getOpenFileName(this, tr("load reference frame normalization Mat ..."),
                                                "/home/jiang/CvTools/DenseOpticalFlow/OpticalFlow_CeLiu/MSresult/", \
                                                tr("Files (*.txt)"));
        refNrmlName = refNrmlfileName.toStdString();
    }
    std::ifstream refNrmlFile(refNrmlName.c_str());
    Eigen::Matrix4f refNrmlMat; refNrmlMat.setZero();
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            refNrmlFile >> refNrmlMat(i, j);
        }
    }
    std::cout<<"loaded ref normalization Mat: "<<refNrmlMat<<"\n";

    // load new sequence
    fileName = QFileDialog::getOpenFileName(this, tr("new scene to merge..."),
                                            "/home/jiang/CvTools/DenseOpticalFlow/OpticalFlow_CeLiu/MSresult/", \
                                            tr("Files (*.pcd)"));

    if(fileName.size()<1) { std::cout<<"PointClouds not added ...\n";  return;  }
    pcl::io::loadPCDFile<PointC>(fileName.toStdString(), *subSeq2);

    // load new sequence normalization mat
    std::string newNrmlName = fileName.toStdString().substr(0, fileName.length() - 4);
    newNrmlName = newNrmlName + "_NrmlMat.txt";
    if(!fexists(newNrmlName.c_str()))
    {
        QString newNrmlfileName;
        newNrmlfileName = QFileDialog::getOpenFileName(this, tr("load newframe normalization Mat ..."),
                                                "/home/jiang/CvTools/DenseOpticalFlow/OpticalFlow_CeLiu/MSresult/", \
                                                tr("Files (*.txt)"));
        newNrmlName = newNrmlfileName.toStdString();
    }
    std::ifstream newNrmlFile(newNrmlName.c_str());
    Eigen::Matrix4f newNrmlMat; newNrmlMat.setZero();
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            newNrmlFile >> newNrmlMat(i, j);
        }
    }
    std::cout<<"loaded new normalization Mat: "<<newNrmlMat<<"\n";

    Eigen::Matrix4f transMatTmp; transMatTmp.setZero();
    // merge point clouds taking the first frame of new sequence as reference
    // 1. inverse last frame to reference frame transformation
    transMatTmp = transMat.inverse();
    pcl::transformPointCloud(*subSeq1, *subSeq1, transMatTmp);
    // 2. inverse reference frame normalization
    transMatTmp = refNrmlMat.inverse();
    pcl::transformPointCloud(*subSeq1, *subSeq1, transMatTmp);
    // 3. impose normalization with the new sequence
    pcl::transformPointCloud(*subSeq1, *subSeq1, newNrmlMat);

    pcl::copyPointCloud(*subSeq1, *str2SceneRansacParams->mergedScene);
    for(int i=0; i<subSeq2->points.size(); i++)
    {
        str2SceneRansacParams->mergedScene->push_back(subSeq2->points.at(i));
    }
    pclVisualizer->removeAllPointClouds();
    displayPointClouds(str2SceneRansacParams->mergedScene,
                       "mergedScene", false);

    std::string mergePcName = fileName.toStdString().substr(0, fileName.length() - 4);
    mergePcName = mergePcName + "_fusions.pcd";
    pcl::io::savePCDFileASCII(mergePcName.c_str(),
                              *str2SceneRansacParams->mergedScene);
}

void KittiVisualizerQt::on_rmBadSeeds_clicked()
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr trkPts(new pcl::PointCloud<pcl::PointXYZ>);
    // load *.pcd file
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"),
                                                    "/home/jiang/CvTools/DenseOpticalFlow/OpticalFlow_CeLiu", tr("Files (*.pcd)"));
    if(fileName.size()<1)
    {
        std::cout<<"trkPts not loaded ...\n";
        return;
    }
    pcl::io::loadPCDFile<pcl::PointXYZ>(fileName.toStdString(), *trkPts);

    PointCloudProcessing pcp;


    colorTrk.reset(new PointCloudC);
    pcl::copyPointCloud(*trkPts, *colorTrk);

    if(colorOpt[0]==0 && colorOpt[1]==0 && colorOpt[2]==0)
    {
        for(int i=0; i<colorTrk->points.size();i++)
        {
            PointC *colorPt = &colorTrk->points.at(i);
            int rgbValue[3] = {0, 0, 0};
            getValueBetweenTwoFixedColors(colorPt->z, rgbValue[0], rgbValue[1], rgbValue[2]);
            colorPt->r = int(rgbValue[0]);
            colorPt->g = int(rgbValue[1]);
            colorPt->b = int(rgbValue[2]);
        }
    }
    else
    {
        for(int i=0; i<colorTrk->points.size();i++)
        {
            PointC *colorPt = &colorTrk->points.at(i);
            colorPt->r = colorOpt[0];
            colorPt->g = colorOpt[1];
            colorPt->b = colorOpt[2];
        }
    }


    pcp.removeBadSeed(colorTrk, searchRadius);
    std::cout<<"seed size after: "<<colorTrk->points.size()<<"\n";

    pclVisualizer->addPointCloud(colorTrk, "colorTrk");
    pclVisualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                    8, "colorTrk");
    ui->qvtkWidget_pclViewer->update();
}

void KittiVisualizerQt::on_regist2RndScenes_clicked()
{

    str2SceneRansacParams->sceneRefName = "/home/jiang/CvTools/DenseOpticalFlow/OpticalFlow_CeLiu/MSresult/people1car1_35_45/FoV3d2d_image_02_0000_45.pcd";
    str2SceneRansacParams->sceneNewName = "/home/jiang/CvTools/DenseOpticalFlow/OpticalFlow_CeLiu/MSresult/people1car1_2exp/FoV3d2d_image_02_0000_56.pcd";
    str2SceneRansacParams->sceneNoMotRef = loadPointClouds(str2SceneRansacParams->sceneRefName);
    str2SceneRansacParams->sceneNoMotNew = loadPointClouds(str2SceneRansacParams->sceneNewName);

    str2SceneRansacParams->motSeedsRefName = "/home/jiang/CvTools/DenseOpticalFlow/OpticalFlow_CeLiu/MSresult/people1car1_35_45/FoV3d2d_image_02_0000_45_mot.pcd";
    str2SceneRansacParams->motSeedsNewName = "/home/jiang/CvTools/DenseOpticalFlow/OpticalFlow_CeLiu/MSresult/people1car1_2exp/FoV3d2d_image_02_0000_56_mot.pcd";
    str2SceneRansacParams->motSeedsRef = loadPointClouds(str2SceneRansacParams->motSeedsRefName);
    str2SceneRansacParams->motSeedsNew = loadPointClouds(str2SceneRansacParams->motSeedsNewName);

    str2SceneRansacParams->corrRefName = "/home/jiang/CvTools/DenseOpticalFlow/OpticalFlow_CeLiu/MSresult/people1car1_45_56/FoV3d2d_image_02_0000_45_features.pcd";
    str2SceneRansacParams->corrNewName = "/home/jiang/CvTools/DenseOpticalFlow/OpticalFlow_CeLiu/MSresult/people1car1_45_56/FoV3d2d_image_02_0000_56_features.pcd";
    str2SceneRansacParams->corrRef = loadPointClouds(str2SceneRansacParams->corrRefName);
    str2SceneRansacParams->corrNew = loadPointClouds(str2SceneRansacParams->corrNewName);

    PointCloudProcessing registrationObj;
    registrationObj.removeBadSeed(str2SceneRansacParams->motSeedsRef, searchRadius);
    registrationObj.pclRegionGrow(str2SceneRansacParams->corrRef,
                                  str2SceneRansacParams->motSeedsRef,
                                  growSpeed, searchRadius, heightThd, \
                                  str2SceneRansacParams->sceneMotRef,
                                  str2SceneRansacParams->motIdxRef);

    registrationObj.removeBadSeed(str2SceneRansacParams->motSeedsNew, searchRadius);
    registrationObj.pclRegionGrow(str2SceneRansacParams->corrNew,
                                  str2SceneRansacParams->motSeedsNew,
                                  growSpeed, searchRadius, heightThd, \
                                  str2SceneRansacParams->sceneMotNew,
                                  str2SceneRansacParams->motIdxNew);
    if(str2SceneRansacParams->motIdxRef.size()>str2SceneRansacParams->motIdxNew.size())
    {
        removeMotions(str2SceneRansacParams->corrRef, str2SceneRansacParams->motIdxRef);
        std::cout<<"corr after motion removal: "<<str2SceneRansacParams->corrRef->points.size()<<"\n";
        removeMotions(str2SceneRansacParams->corrNew, str2SceneRansacParams->motIdxRef);
        std::cout<<"corr after motion removal: "<<str2SceneRansacParams->corrNew->points.size()<<"\n";
    }else
    {
        removeMotions(str2SceneRansacParams->corrRef, str2SceneRansacParams->motIdxRef);
        std::cout<<"corr after motion removal: "<<str2SceneRansacParams->corrRef->points.size()<<"\n";
        removeMotions(str2SceneRansacParams->corrNew, str2SceneRansacParams->motIdxRef);
        std::cout<<"corr after motion removal: "<<str2SceneRansacParams->corrNew->points.size()<<"\n";
    }
    registrationObj.normalizePointClouds(str2SceneRansacParams->corrRef,
                                         str2SceneRansacParams->normalizationMat);
    pcl::transformPointCloud(*str2SceneRansacParams->sceneNoMotRef,
                             *str2SceneRansacParams->sceneNoMotRef,
                             str2SceneRansacParams->normalizationMat);
    pcl::transformPointCloud(*str2SceneRansacParams->sceneNoMotNew,
                             *str2SceneRansacParams->sceneNoMotNew,
                             str2SceneRansacParams->normalizationMat);
    pcl::transformPointCloud(*str2SceneRansacParams->corrNew,
                             *str2SceneRansacParams->corrNew,
                             str2SceneRansacParams->normalizationMat);

    // remove preprocessing data visualization
    pclVisualizer->removePointCloud(str2SceneRansacParams->sceneRefName.c_str());
    pclVisualizer->removePointCloud(str2SceneRansacParams->sceneNewName.c_str());
    pclVisualizer->removePointCloud(str2SceneRansacParams->corrRefName.c_str());
    pclVisualizer->removePointCloud(str2SceneRansacParams->corrNewName.c_str());
    pclVisualizer->removePointCloud(str2SceneRansacParams->motSeedsRefName.c_str());
    pclVisualizer->removePointCloud(str2SceneRansacParams->motSeedsNewName.c_str());
    pclVisualizer->removePointCloud(str2SceneRansacParams->registeredName.c_str());
    // register 2 scenes
    std::cout<<"start 2 scenes registration...\n";
    registrationObj.register2ScenesRansac(str2SceneRansacParams->sceneNoMotRef,
                                          str2SceneRansacParams->sceneNoMotNew,
                                          str2SceneRansacParams->corrRef,
                                          str2SceneRansacParams->corrNew,
                                          str2SceneRansacParams->inlrThdRansac,
                                          str2SceneRansacParams->sampleNbRansac,
                                          str2SceneRansacParams->inlrRateRansac,
                                          str2SceneRansacParams->maxIterRansac,
                                          str2SceneRansacParams->transMat,
                                          str2SceneRansacParams->registeredScene);
    displayPointClouds(str2SceneRansacParams->registeredScene,
                       str2SceneRansacParams->registeredName.c_str(), false);
//    displayPointClouds(str2SceneRansacParams->sceneNoMotRef,
//                       str2SceneRansacParams->sceneRefName.c_str(), false);
//    displayPointClouds(str2SceneRansacParams->corrRef,
//                       str2SceneRansacParams->corrRefName.c_str(), 8);
//    displayPointClouds(str2SceneRansacParams->sceneNoMotNew,
//                       str2SceneRansacParams->sceneNewName.c_str(), false);
//    displayPointClouds(str2SceneRansacParams->corrNew,
//                       str2SceneRansacParams->corrNewName.c_str(), 8);

}

void KittiVisualizerQt::on_crossMergeSubSeq_clicked()
{

}
