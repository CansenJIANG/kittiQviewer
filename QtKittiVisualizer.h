#ifndef QT_KITTI_VISUALIZER_H
#define QT_KITTI_VISUALIZER_H

#include <vector>
// Qt
#include <QMainWindow>
#include <QWidget>

// Boost
#include <boost/program_options.hpp>

// PCL
#include <pcl/visualization/keyboard_event.h>
#include <pcl/visualization/pcl_visualizer.h>

// VTK
#include <vtkRenderWindow.h>

#include "KittiDataset.h"
#include "pointcloudprocessing.h"

#include <kitti-devkit-raw/tracklets.h>

// Qt header
#include <QString>
#include <QtDebug>
#include <QMainWindow>
#include <QDialog>
#include <QFileDialog>
#include <QTextEdit>
#include <QPlainTextEdit>
#include <QtCore/QEvent>
#include <QtGui/QKeyEvent>

namespace Ui
{
class KittiVisualizerQt;
}

struct str_2ScenesRansac
{
    // 2 scenes registration data
    PointCloudT::Ptr sceneRef;
    PointCloudT::Ptr sceneNew;
    PointCloudT::Ptr corrRef;
    PointCloudT::Ptr corrNew;
    PointCloudT::Ptr motSeedsRef;
    PointCloudT::Ptr motSeedsNew;
    PointCloudT::Ptr sceneMotRef;
    PointCloudT::Ptr sceneMotNew;
    PointCloudT::Ptr sceneNoMotRef;
    PointCloudT::Ptr sceneNoMotNew;
    PointCloudT::Ptr registeredScene;

    // point cloud display names
    std::string sceneRefName;
    std::string sceneNewName;
    std::string corrRefName;
    std::string corrNewName;
    std::string motSeedsRefName;
    std::string motSeedsNewName;
    std::string registeredName;

    // moving object indices
    std::set<int> motIdxRef;
    std::set<int> motIdxNew;

    // Ransac settings
    float inlrRateRansac;
    float smpRateRansac;
    float inlrThdRansac;
    int   maxIterRansac;
    int   sampleNbRansac;

    // estimated rigid transformation
    Eigen::Matrix4f transMat;
    Eigen::Matrix4f normalizationMat;
};

class KittiVisualizerQt : public QMainWindow
{
    Q_OBJECT

public:

    KittiVisualizerQt(QWidget *parent, int argc, char** argv);
    virtual ~KittiVisualizerQt();

    bool loadDataset();
    bool loadNextFrame();
    bool loadPreviousFrame();

    void getTrackletColor(const KittiTracklet& tracklet, int &r, int& g, int& b);
    PointCloudT::Ptr loadPointClouds(std::string &filesName);
    void removeMotions(PointCloudT::Ptr &scene, std::set<int> clusterIdx);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr
    displayPointClouds(PointCloudT::Ptr inputClouds, std::string cloudName);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr
    displayPointClouds(PointCloudT::Ptr inputClouds, std::string cloudName, int ptSize);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr
    displayPointClouds(PointCloudT::Ptr inputClouds, std::string cloudName, int ptColor[]);

public slots:

    void newDatasetRequested(int value);
    void newFrameRequested(int value);
    void newTrackletRequested(int value);

    void showFramePointCloudToggled(bool value);
    void showTrackletBoundingBoxesToggled(bool value);
    void showTrackletPointCloudsToggled(bool value);
    void showTrackletInCenterToggled(bool value);

private slots:
    void on_loadPc_clicked();

    void on_loadTrkPts_clicked();

    void on_loadTrkPtColor_clicked();

    void on_clearTrkPts_clicked();

    void on_addPc_clicked();

    void on_deletePc_clicked();

    void on_colorOpt_activated(const QString &arg1);

    void on_colorOpt_currentIndexChanged(int index);

    void on_addTrkPts_clicked();

    void on_segMovObj_clicked();

    void on_regionGrowR_editingFinished();

    void on_heightThd_editingFinished();

    void on_growSpeed_editingFinished();

    void on_removeMot_clicked();

    void on_frontView_clicked();

    void on_FoV3d2d_clicked();

    void on_addFoV3d2d_clicked();

    void on_register2scenes_clicked();

    void on_load2scenes_clicked();

    void on_inlrRateRansac_editingFinished();

    void on_smpRateRansac_editingFinished();

    void on_inlrThdRansac_editingFinished();

    void on_maxIterRansac_editingFinished();

    void on_showMot_clicked();

    void on_loadRefScene_clicked();

    void on_addRegistScene_clicked();

private:

    int parseCommandLineOptions(int argc, char** argv);

    int dataset_index;
    KittiDataset* dataset;

    int frame_index;

    int tracklet_index;

    pcl::visualization::PCLVisualizer::Ptr pclVisualizer;

    void loadAvailableTracklets();
    void clearAvailableTracklets();
    std::vector<KittiTracklet> availableTracklets;

    void updateDatasetLabel();
    void updateFrameLabel();
    void updateTrackletLabel();

    void loadPointCloud();
    void showPointCloud();
    void hidePointCloud();
    bool pointCloudVisible;
    KittiPointCloud::Ptr pointCloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorCloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorTrk;
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene;
    pcl::PointCloud<pcl::PointXYZ>::Ptr seeds;
    float searchRadius;
    float heightThd;
    float growSpeed;

    str_2ScenesRansac *str2SceneRansacParams;

    std::set<int> clusterIdx;
    int colorOpt[3];

    void showTrackletBoxes();
    void hideTrackletBoxes();
    bool trackletBoundingBoxesVisible;

    void loadTrackletPoints();
    void showTrackletPoints();
    void hideTrackletPoints();
    void clearTrackletPoints();
    bool trackletPointsVisible;
    std::vector<KittiPointCloud::Ptr> croppedTrackletPointClouds;

    void showTrackletInCenter();
    void hideTrackletInCenter();
    bool trackletInCenterVisible;

    void setFrameNumber(int frameNumber);

    void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event,
                                void* viewer_void);

    Ui::KittiVisualizerQt *ui;

};

#endif // QT_KITTI_VISUALIZER_H
