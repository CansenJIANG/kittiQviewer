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
struct str_displayMotions
{
  float runMotFps;
  std::vector<PointCloudC> nMots;
};
struct str_2ScenesRansac
{
  // 2 scenes registration data
  PointCloudC::Ptr sceneRef;
  PointCloudC::Ptr sceneNew;
  PointCloudC::Ptr corrRef;
  PointCloudC::Ptr corrNew;
  PointCloudC::Ptr motSeedsRef;
  PointCloudC::Ptr motSeedsNew;
  PointCloudC::Ptr sceneMotRef;
  PointCloudC::Ptr sceneMotNew;
  PointCloudC::Ptr sceneNoMotRef;
  PointCloudC::Ptr sceneNoMotNew;
  PointCloudC::Ptr registeredScene;
  PointCloudC::Ptr mergedScene;

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
  bool rmMotState;

  // Ransac settings
  float inlrRateRansac;
  float smpRateRansac;
  float inlrThdRansac;
  int   maxIterRansac;
  int   sampleNbRansac;
  int registLength;

  // estimated rigid transformation
  Eigen::Matrix4f transMat;
  Eigen::Matrix4f normalizationMat;
};

struct str_DennisParam
{
  PointCloudC::Ptr cloud;
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

  void init();
  void getTrackletColor(const KittiTracklet& tracklet, int &r, int& g, int& b);
  // load point cloud
  PointCloudC::Ptr loadPointClouds(std::string &filesName);
  // remove motions
  void removeMotions(PointCloudC::Ptr &scene, std::set<int> clusterIdx);

  // display point cloud
  PointCloudC::Ptr
  displayPointClouds(PointCloudC::Ptr inputClouds, std::string cloudName, bool heightColor);

  // remove motions and retrun segmented part
  void removeMotions(PointCloudC::Ptr &sceneIn,
                     std::set<int> clusterIdx,
                     PointCloudC::Ptr &motsOut);
  PointCloudC::Ptr
  displayPointClouds(PointCloudC::Ptr inputClouds, std::string cloudName, int ptSize);
  PointCloudC::Ptr
  displayPointClouds(PointCloudC::Ptr inputClouds, std::string cloudName, int ptColor[]);
  void voxelDensityFiltering(PointCloudC::Ptr& registeredScene);

  QString loadNmots(std::vector<PointCloudC> & motSeq);
  QString loadNmots(std::vector<PointCloudC> & motSeq, const std::string fileExt);
  QString loadNmots(std::vector<PointCloudC> & motSeq, const std::string fileExt, int colorIdx);


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
  void on_segNmots_clicked();

  void on_regionGrowR_editingFinished();

  void on_heightThd_editingFinished();

  void on_growSpeed_editingFinished();

  void on_removeMot_clicked();

  void on_loadColorCloud_clicked();

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

  void on_nScenesRansac_clicked();

  void on_saveRegist_clicked();

  void on_seqRegist_clicked();

  void on_mergeSubSeq_clicked();

  void on_rmBadSeeds_clicked();

  void on_regist2RndScenes_clicked();

  void on_crossMergeSubSeq_clicked();

  void on_removeMotState_stateChanged(int arg1);

  void on_algoReset_clicked();

  void on_showAxis_stateChanged(int arg1);

  void on_seqRegistMots_clicked();

  void on_seqRegistAll_clicked();

  void on_ransac2framesRegist_clicked();

  void on_mapTexture_clicked();

  void on_ransacMS_clicked();

  void on_ransac2framesRmBkg_clicked();

  void on_rmNMots_clicked();

  void on_addClouds_clicked();

  void on_addFwdPts_clicked();

  void on_addBwdPts_clicked();

  void on_seqRegistNmotsRunning_clicked();

  void on_runMots_clicked();

  void on_runMotFps_editingFinished();

  void on_loadNmots_clicked();

  void on_fuseMotsSeq_clicked();

  void on_loadMotsFusion_clicked();

  void on_stationNmots_clicked();

  void on_junctionNmots_clicked();

  void on_junctionLoadMots_clicked();

  void on_displayMots_clicked();

  //////////////////////////////////////////////////////////////////////////
  /// SECTION FOR DENNIS
  ///
  //////////////////////////////////////////////////////////////////////////
  void on_dennis_loadKitti3D_clicked(); // load the 3D data from KITTI dataset (.bin file)

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
  PointCloudC::Ptr colorCloud;
  PointCloudC::Ptr colorTrk;
  PointCloudC::Ptr scene;
  PointCloudC::Ptr seeds;
  float searchRadius;
  float heightThd;
  float growSpeed;

  str_2ScenesRansac *str2SceneRansacParams;
  str_displayMotions *strDisplayMotions;

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

  //////////////////////////////////////////////////////////////////////////
  /// SECTION FOR DENNIS
  ///
  //////////////////////////////////////////////////////////////////////////
  str_DennisParam *strDennisParam; // create the structure for variables' control

};

#endif // QT_KITTI_VISUALIZER_H
