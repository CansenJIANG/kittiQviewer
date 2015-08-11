#include "QtKittiVisualizer.h"
#include "ui_kittiQviewer.h"
#include <fstream>
#include <string>

#include <QCheckBox>
#include <QLabel>
#include <QMainWindow>
#include <QSlider>
#include <QWidget>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <eigen3/Eigen/Core>

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
    colorCloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    colorTrk.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    scene.reset(new PointCloudT);
    seeds.reset(new PointCloudT);
    searchRadius = 0.8;
    heightThd = -1.5;
    growSpeed = 0.5;
    clusterIdx.clear();

    // 2 scenes registration data
    str2SceneRansacParams = new str_2ScenesRansac;
    str2SceneRansacParams->sceneRef.reset(new PointCloudT);
    str2SceneRansacParams->sceneNew.reset(new PointCloudT);
    str2SceneRansacParams->corrRef .reset(new PointCloudT);
    str2SceneRansacParams->corrNew .reset(new PointCloudT);
    str2SceneRansacParams->motSeedsRef.reset(new PointCloudT);
    str2SceneRansacParams->motSeedsNew.reset(new PointCloudT);
    str2SceneRansacParams->sceneMotRef  .reset(new PointCloudT);
    str2SceneRansacParams->sceneMotNew  .reset(new PointCloudT);
    str2SceneRansacParams->sceneNoMotRef.reset(new PointCloudT);
    str2SceneRansacParams->sceneNoMotNew.reset(new PointCloudT);
    str2SceneRansacParams->registeredScene.reset(new PointCloudT);

    // point cloud display names
    str2SceneRansacParams->sceneRefName    = "load Reference Scene";
    str2SceneRansacParams->sceneNewName    = "load New Scene";
    str2SceneRansacParams->corrRefName     = "load Reference Correspondences";
    str2SceneRansacParams->corrNewName     = "load New Correspondences";
    str2SceneRansacParams->motSeedsRefName = "load Reference Motion Seeds";
    str2SceneRansacParams->motSeedsNewName = "load New Motion Seeds";
    str2SceneRansacParams->registeredName  = "registeredCloud";

    // Ransac parameter settings
    str2SceneRansacParams->inlrThdRansac = 0.02;
    str2SceneRansacParams->smpRateRansac = 0.7;
    str2SceneRansacParams->inlrRateRansac = 0.8;
    str2SceneRansacParams->maxIterRansac = 500;
    str2SceneRansacParams->sampleNbRansac = 3;
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
    pcl::PointXYZRGB min_pt, max_pt;
    pcl::getMinMax3D(*colorCloud, min_pt, max_pt);
//    float minMax = max_pt.z - min_pt.z;
//    std::cout<<"max_pt.z: "<<max_pt.z<<", min_pt.z"<<min_pt.z;
    float minMax = max_pt.z + 2.0;
    for(int i=0; i<colorCloud->points.size();i++)
    {
        pcl::PointXYZRGB *colorPt = &colorCloud->points.at(i);
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

pcl::PointCloud<pcl::PointXYZRGB>::Ptr
KittiVisualizerQt::displayPointClouds(PointCloudT::Ptr inputClouds, std::string cloudName)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputCloudsColor
            (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*inputClouds, *inputCloudsColor);

    for(int i=0; i<inputCloudsColor->points.size(); i++)
    {
        pcl::PointXYZRGB *point = &inputCloudsColor->points.at(i);
        if( (*point).z <-2.5)
        {
            (*point).x = 0; (*point).y = 0; (*point).z = 0;
        }
    }
    // get minimum and maximum
    pcl::PointXYZRGB min_pt, max_pt;
    pcl::getMinMax3D(*inputCloudsColor, min_pt, max_pt);
    float minMax = max_pt.z + 2.0;
    for(int i=0; i<inputCloudsColor->points.size();i++)
    {
        pcl::PointXYZRGB *colorPt = &inputCloudsColor->points.at(i);
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
    ui->qvtkWidget_pclViewer->update();
    return inputCloudsColor;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr
KittiVisualizerQt::displayPointClouds(PointCloudT::Ptr inputClouds, std::string cloudName, int ptSize)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputCloudsColor
            (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*inputClouds, *inputCloudsColor);

    for(int i=0; i<inputCloudsColor->points.size(); i++)
    {
        pcl::PointXYZRGB *point = &inputCloudsColor->points.at(i);
        if( (*point).z <-2.5)
        {
            (*point).x = 0; (*point).y = 0; (*point).z = 0;
        }
    }
    // get minimum and maximum
    pcl::PointXYZRGB min_pt, max_pt;
    pcl::getMinMax3D(*inputCloudsColor, min_pt, max_pt);
    float minMax = max_pt.z + 2.0;
    for(int i=0; i<inputCloudsColor->points.size();i++)
    {
        pcl::PointXYZRGB *colorPt = &inputCloudsColor->points.at(i);
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

pcl::PointCloud<pcl::PointXYZRGB>::Ptr
KittiVisualizerQt::displayPointClouds(PointCloudT::Ptr inputClouds, std::string cloudName, int ptColor[3])
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputCloudsColor
            (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*inputClouds, *inputCloudsColor);

    for(int i=0; i<inputCloudsColor->points.size(); i++)
    {
        pcl::PointXYZRGB *point = &inputCloudsColor->points.at(i);
        if( (*point).z <-2.5)
        {
            (*point).x = 0; (*point).y = 0; (*point).z = 0;
        }
    }
    // get minimum and maximum

    for(int i=0; i<inputCloudsColor->points.size();i++)
    {
        pcl::PointXYZRGB *colorPt = &inputCloudsColor->points.at(i);
        colorPt->r = ptColor[0];
        colorPt->g = ptColor[1];
        colorPt->b = ptColor[2];
    }
    //    showPointCloud();
    pclVisualizer->removePointCloud(cloudName.c_str());
    pclVisualizer->addPointCloud(inputCloudsColor, cloudName.c_str());
    pclVisualizer->setPointCloudRenderingProperties(
                    pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, cloudName.c_str());

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

    colorTrk.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*trkPts, *colorTrk);
    if(colorOpt[0]==0 && colorOpt[1]==0 && colorOpt[2]==0)
    {
        for(int i=0; i<colorTrk->points.size();i++)
        {
            pcl::PointXYZRGB *colorPt = &colorTrk->points.at(i);
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
            pcl::PointXYZRGB *colorPt = &colorTrk->points.at(i);
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
        pcl::PointXYZRGB *colorPt = &colorTrk->points.at(i);
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

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*pointCloud, *colorCloud);
    // get minimum and maximum
    pcl::PointXYZRGB min_pt, max_pt;
    pcl::getMinMax3D(*colorCloud, min_pt, max_pt);
    float minMax = max_pt.z - min_pt.z;
//    std::cout<<"max_pt.z: "<<max_pt.z<<", min_pt.z"<<min_pt.z;
    for(int i=0; i<colorCloud->points.size();i++)
    {
        pcl::PointXYZRGB *colorPt = &colorCloud->points.at(i);
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


    pcl::PointCloud<pcl::PointXYZRGB>::Ptr trkPtsColor(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*trkPts, *trkPtsColor);
    if(colorOpt[0]==0 && colorOpt[1]==0 && colorOpt[2]==0)
    {
        for(int i=0; i<trkPtsColor->points.size();i++)
        {
            pcl::PointXYZRGB *colorPt = &trkPtsColor->points.at(i);
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
            pcl::PointXYZRGB *colorPt = &trkPtsColor->points.at(i);
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
    PointCloudT::Ptr cloudSeg(new PointCloudT);
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
        pcl::io::loadPCDFile<pcl::PointXYZ>(fileName.toStdString(), *scene);
        colorCloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        colorCloud = displayPointClouds(scene, "newPc");
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
        pcl::io::loadPCDFile<pcl::PointXYZ>(fileName.toStdString(), *seeds);
        pcl::copyPointCloud(*seeds,*colorTrk);
    }

    PointCloudProcessing segMot;
    std::cout<<"start region growing...\n";
    segMot.pclRegionGrow(scene, seeds, growSpeed, searchRadius, heightThd, cloudSeg, clusterIdx);
    std::cout<<"cluster size: "<<cloudSeg->points.size();
    pclVisualizer->removePointCloud("segMot");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloudSegCH(cloudSeg, 255, 0, 0);
    pclVisualizer->addPointCloud<pcl::PointXYZ>(cloudSeg, cloudSegCH, "segMot");
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
        pcl::PointXYZRGB *pointMot = &colorCloud->points.at(*clusterIdxIter);
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
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr motColor(new pcl::PointCloud<pcl::PointXYZRGB>);
    if(clusterIdx.size()<1) {   return;    }
    if(colorCloud->size()<1){return;}
    std::set<int>::iterator clusterIdxIter = clusterIdx.begin();
    for(int i=0; i<clusterIdx.size(); i++, clusterIdxIter++)
    {
        pcl::PointXYZRGB pointMot = colorCloud->points.at(*clusterIdxIter);
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


void KittiVisualizerQt::removeMotions(PointCloudT::Ptr &scene, std::set<int> clusterIdx)
{
    if(clusterIdx.size()<1) {   return;    }
    std::set<int>::iterator clusterIdxIter = clusterIdx.end();
    while(clusterIdxIter!=clusterIdx.begin())
    {
        --clusterIdxIter;
        PointT *pointMot = &scene->points.at(*clusterIdxIter);
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
        pcl::PointXYZRGB *point = &colorCloud->points.at(i);
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
    pcl::io::loadPCDFile<pcl::PointXYZ>(fileName.toStdString(), *scene);
    for(int i=0; i<scene->points.size(); i++)
    {
        PointT *point = &scene->points.at(i);
        if( (*point).z <-2.5)
        {
            (*point).x = 0; (*point).y = 0; (*point).z = 0;
        }
    }
    pcl::copyPointCloud(*scene, *colorCloud);
    pcl::copyPointCloud(*scene, *pointCloud);
    // get minimum and maximum
    pcl::PointXYZRGB min_pt, max_pt;
    pcl::getMinMax3D(*colorCloud, min_pt, max_pt);
    float minMax = max_pt.z + 2.0;
    for(int i=0; i<colorCloud->points.size();i++)
    {
        pcl::PointXYZRGB *colorPt = &colorCloud->points.at(i);
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
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr addFoV3d2d(new pcl::PointCloud<pcl::PointXYZRGB>);
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"),
                       "/home/jiang/CvTools/DenseOpticalFlow/OpticalFlow_CeLiu", \
                                                    tr("Files (*.pcd)"));

    if(fileName.size()<1)
    {
        std::cout<<"FoV3d2d point cloud not added ...\n";
        return;
    }
    addFoV3d2d->points.clear();
    pcl::io::loadPCDFile<pcl::PointXYZRGB>(fileName.toStdString(), *addFoV3d2d);
    for(int i=0; i<addFoV3d2d->points.size(); i++)
    {
        pcl::PointXYZRGB *point = &addFoV3d2d->points.at(i);
        if( (*point).z <-2.5)
        {
            (*point).x = 0; (*point).y = 0; (*point).z = 0;
        }
    }

    // get minimum and maximum
    pcl::PointXYZRGB min_pt, max_pt;
    pcl::getMinMax3D(*addFoV3d2d, min_pt, max_pt);
    float minMax = max_pt.z + 2.0;
    for(int i=0; i<addFoV3d2d->points.size();i++)
    {
        pcl::PointXYZRGB *colorPt = &addFoV3d2d->points.at(i);
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
PointCloudT::Ptr
KittiVisualizerQt::loadPointClouds(std::string &filesName)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr addFoV3d2d(new pcl::PointCloud<pcl::PointXYZRGB>);
    PointCloudT::Ptr addFoV3d2dNoColor(new PointCloudT);
    QString fileName;
    if(!fexists(filesName.c_str()))
    {
        fileName = QFileDialog::getOpenFileName(this, tr(filesName.c_str()),
                           "/home/jiang/CvTools/DenseOpticalFlow/OpticalFlow_CeLiu/MSresult/people1car1_2exp/", \
                                                        tr("Files (*.pcd)"));

        if(fileName.size()<1)
        {
            std::cout<<"PointClouds not added ...\n";
            return addFoV3d2dNoColor;
        }
    }else{fileName = QString(filesName.c_str());}

    addFoV3d2d->points.clear();
    pcl::io::loadPCDFile<pcl::PointXYZRGB>(fileName.toStdString(), *addFoV3d2d);
    int fileNamelen = fileName.toStdString().length();
    std::string textureName = fileName.toStdString().substr(0, fileNamelen-4) + "_texture.pcd";
    std::cout<<"textureName: "<<textureName<<"\n";
    if(!fexists(textureName.c_str()))
    {
        std::cout<<"Scene texture not exist, mapping color with height ...\n";
        for(int i=0; i<addFoV3d2d->points.size(); i++)
        {
            pcl::PointXYZRGB *point = &addFoV3d2d->points.at(i);
            if( (*point).z <-2.5)
            {
                (*point).x = 0; (*point).y = 0; (*point).z = 0;
            }
        }

        // get minimum and maximum
        pcl::PointXYZRGB min_pt, max_pt;
        pcl::getMinMax3D(*addFoV3d2d, min_pt, max_pt);
        float minMax = max_pt.z + 2.0;
        for(int i=0; i<addFoV3d2d->points.size();i++)
        {
            pcl::PointXYZRGB *colorPt = &addFoV3d2d->points.at(i);
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
            pcl::PointXYZRGB *colorPt = &addFoV3d2d->points.at(i);
            PointT ptTexture = cloudTexture->points.at(i);
            colorPt->r = ptTexture.x;
            colorPt->g = ptTexture.y;
            colorPt->b = ptTexture.z;
        }
    }
    std::cout<<"addFoV3d2d size: "<< addFoV3d2d->points.size()<<std::endl;
    pclVisualizer->addPointCloud(addFoV3d2d, fileName.toStdString().c_str());
    std::cout<<"loaded: "<<fileName.toStdString().c_str()<<"\n";
    ui->qvtkWidget_pclViewer->update();

    filesName.clear(); filesName = fileName.toStdString();
    pcl::copyPointCloud(*addFoV3d2d, *addFoV3d2dNoColor);
    return addFoV3d2dNoColor;
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
    registrationObj.pclRegionGrow(str2SceneRansacParams->sceneRef,
                                  str2SceneRansacParams->motSeedsRef,
                                  growSpeed, searchRadius, heightThd, \
                                  str2SceneRansacParams->sceneMotRef,
                                  str2SceneRansacParams->motIdxRef);
//    std::cout<<"size of ref scene: "<<str2SceneRansacParams->sceneRef->points.size()<<"\n";
    removeMotions(str2SceneRansacParams->sceneNoMotRef, str2SceneRansacParams->motIdxRef);
//    std::cout<<"size of ref scene no motion: "<<str2SceneRansacParams->sceneNoMotRef->points.size()<<"\n";

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
                      str2SceneRansacParams->sceneRefName);
    displayPointClouds(str2SceneRansacParams->sceneNoMotNew,
                      str2SceneRansacParams->sceneNewName);

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
                       str2SceneRansacParams->registeredName.c_str());
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
    str2SceneRansacParams->corrRefName = basedDir + "_segMot_bkg.pcd";
    str2SceneRansacParams->motSeedsRefName = basedDir + "_segMot_Motions.pcd";

    str2SceneRansacParams->corrRef  = loadPointClouds(str2SceneRansacParams->corrRefName);
    str2SceneRansacParams->motSeedsRef = loadPointClouds(str2SceneRansacParams->motSeedsRefName);
    pcl::copyPointCloud(*(str2SceneRansacParams->sceneRef),
                        *(str2SceneRansacParams->sceneNoMotRef));

    // motion segmention
    PointCloudProcessing registrationObj;
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
                      str2SceneRansacParams->sceneRefName);

    displayPointClouds(str2SceneRansacParams->corrRef, str2SceneRansacParams->corrRefName, 10);

}

void KittiVisualizerQt::on_addRegistScene_clicked()
{
    str2SceneRansacParams->sceneNewName     = "load New Scene";
    str2SceneRansacParams->corrNewName      = "load New Correspondences";
    str2SceneRansacParams->motSeedsNewName  = "load New Motion Seeds";
    str2SceneRansacParams->sceneNew = loadPointClouds(str2SceneRansacParams->sceneNewName);
    str2SceneRansacParams->corrNew = loadPointClouds(str2SceneRansacParams->corrNewName);
    str2SceneRansacParams->motSeedsNew = loadPointClouds(str2SceneRansacParams->motSeedsNewName);
    pcl::copyPointCloud(*(str2SceneRansacParams->sceneNew),
                        *(str2SceneRansacParams->sceneNoMotNew));

    PointCloudProcessing registrationObj;
    registrationObj.pclRegionGrow(str2SceneRansacParams->sceneNew,
                                  str2SceneRansacParams->motSeedsNew,
                                  growSpeed, searchRadius, heightThd, \
                                  str2SceneRansacParams->sceneMotNew,
                                  str2SceneRansacParams->motIdxNew);

    removeMotions(str2SceneRansacParams->sceneNoMotNew, str2SceneRansacParams->motIdxNew);
    pclVisualizer->removePointCloud(str2SceneRansacParams->sceneNewName.c_str());
    pclVisualizer->removePointCloud(str2SceneRansacParams->corrNewName.c_str());
    pclVisualizer->removePointCloud(str2SceneRansacParams->motSeedsNewName.c_str());

    pcl::transformPointCloud(*str2SceneRansacParams->corrNew, *str2SceneRansacParams->corrNew,
                             str2SceneRansacParams->normalizationMat);
    pcl::transformPointCloud(*str2SceneRansacParams->sceneNoMotNew, *str2SceneRansacParams->sceneNoMotNew,
                             str2SceneRansacParams->normalizationMat);

    displayPointClouds(str2SceneRansacParams->sceneNoMotNew,
                      str2SceneRansacParams->sceneNewName);

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
                       str2SceneRansacParams->registeredName.c_str());
    pcl::transformPointCloud(*str2SceneRansacParams->corrNew, *str2SceneRansacParams->corrNew,
                             str2SceneRansacParams->transMat);
    int ptColorRef[3] = {255, 0, 0};
    int ptColorNew[3] = {0, 255, 0};
    displayPointClouds(str2SceneRansacParams->corrRef, str2SceneRansacParams->corrRefName, ptColorRef);
    displayPointClouds(str2SceneRansacParams->corrNew, str2SceneRansacParams->corrNewName, ptColorNew);
    ui->qvtkWidget_pclViewer->update();
}
