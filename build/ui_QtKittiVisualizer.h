/********************************************************************************
** Form generated from reading UI file 'QtKittiVisualizer.ui'
**
** Created by: Qt User Interface Compiler version 4.8.6
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_QTKITTIVISUALIZER_H
#define UI_QTKITTIVISUALIZER_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QMainWindow>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QSlider>
#include <QtGui/QSplitter>
#include <QtGui/QStatusBar>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>
#include "QVTKWidget.h"

QT_BEGIN_NAMESPACE

class Ui_KittiVisualizerQt
{
public:
    QAction *actionExit;
    QWidget *centralwidget;
    QHBoxLayout *horizontalLayout_2;
    QSplitter *splitterVertical;
    QSplitter *splitterHorizontal;
    QWidget *layoutWidget;
    QVBoxLayout *verticalLayout_sliders;
    QVBoxLayout *verticalLayout_dataSet;
    QLabel *label_dataSet;
    QSlider *slider_dataSet;
    QVBoxLayout *verticalLayout_frame;
    QLabel *label_frame;
    QSlider *slider_frame;
    QVBoxLayout *verticalLayout_tracklet;
    QLabel *label_tracklet;
    QSlider *slider_tracklet;
    QWidget *layoutWidget1;
    QVBoxLayout *verticalLayout_checkBoxes;
    QCheckBox *checkBox_showFramePointCloud;
    QCheckBox *checkBox_showTrackletBoundingBoxes;
    QCheckBox *checkBox_showTrackletPointClouds;
    QCheckBox *checkBox_showTrackletInCenter;
    QVTKWidget *qvtkWidget_pclViewer;
    QStatusBar *statusBar;
    QMenuBar *menuBar;
    QMenu *menuFile;

    void setupUi(QMainWindow *KittiVisualizerQt)
    {
        if (KittiVisualizerQt->objectName().isEmpty())
            KittiVisualizerQt->setObjectName(QString::fromUtf8("KittiVisualizerQt"));
        KittiVisualizerQt->resize(989, 503);
        KittiVisualizerQt->setMinimumSize(QSize(0, 0));
        KittiVisualizerQt->setMaximumSize(QSize(5000, 5000));
        actionExit = new QAction(KittiVisualizerQt);
        actionExit->setObjectName(QString::fromUtf8("actionExit"));
        centralwidget = new QWidget(KittiVisualizerQt);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        horizontalLayout_2 = new QHBoxLayout(centralwidget);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        splitterVertical = new QSplitter(centralwidget);
        splitterVertical->setObjectName(QString::fromUtf8("splitterVertical"));
        splitterVertical->setOrientation(Qt::Horizontal);
        splitterHorizontal = new QSplitter(splitterVertical);
        splitterHorizontal->setObjectName(QString::fromUtf8("splitterHorizontal"));
        splitterHorizontal->setOrientation(Qt::Vertical);
        layoutWidget = new QWidget(splitterHorizontal);
        layoutWidget->setObjectName(QString::fromUtf8("layoutWidget"));
        verticalLayout_sliders = new QVBoxLayout(layoutWidget);
        verticalLayout_sliders->setObjectName(QString::fromUtf8("verticalLayout_sliders"));
        verticalLayout_sliders->setContentsMargins(0, 0, 0, 0);
        verticalLayout_dataSet = new QVBoxLayout();
        verticalLayout_dataSet->setObjectName(QString::fromUtf8("verticalLayout_dataSet"));
        label_dataSet = new QLabel(layoutWidget);
        label_dataSet->setObjectName(QString::fromUtf8("label_dataSet"));

        verticalLayout_dataSet->addWidget(label_dataSet);

        slider_dataSet = new QSlider(layoutWidget);
        slider_dataSet->setObjectName(QString::fromUtf8("slider_dataSet"));
        slider_dataSet->setOrientation(Qt::Horizontal);

        verticalLayout_dataSet->addWidget(slider_dataSet);


        verticalLayout_sliders->addLayout(verticalLayout_dataSet);

        verticalLayout_frame = new QVBoxLayout();
        verticalLayout_frame->setObjectName(QString::fromUtf8("verticalLayout_frame"));
        label_frame = new QLabel(layoutWidget);
        label_frame->setObjectName(QString::fromUtf8("label_frame"));

        verticalLayout_frame->addWidget(label_frame);

        slider_frame = new QSlider(layoutWidget);
        slider_frame->setObjectName(QString::fromUtf8("slider_frame"));
        slider_frame->setOrientation(Qt::Horizontal);

        verticalLayout_frame->addWidget(slider_frame);


        verticalLayout_sliders->addLayout(verticalLayout_frame);

        verticalLayout_tracklet = new QVBoxLayout();
        verticalLayout_tracklet->setObjectName(QString::fromUtf8("verticalLayout_tracklet"));
        label_tracklet = new QLabel(layoutWidget);
        label_tracklet->setObjectName(QString::fromUtf8("label_tracklet"));

        verticalLayout_tracklet->addWidget(label_tracklet);

        slider_tracklet = new QSlider(layoutWidget);
        slider_tracklet->setObjectName(QString::fromUtf8("slider_tracklet"));
        slider_tracklet->setOrientation(Qt::Horizontal);

        verticalLayout_tracklet->addWidget(slider_tracklet);


        verticalLayout_sliders->addLayout(verticalLayout_tracklet);

        splitterHorizontal->addWidget(layoutWidget);
        layoutWidget1 = new QWidget(splitterHorizontal);
        layoutWidget1->setObjectName(QString::fromUtf8("layoutWidget1"));
        verticalLayout_checkBoxes = new QVBoxLayout(layoutWidget1);
        verticalLayout_checkBoxes->setObjectName(QString::fromUtf8("verticalLayout_checkBoxes"));
        verticalLayout_checkBoxes->setContentsMargins(0, 0, 0, 0);
        checkBox_showFramePointCloud = new QCheckBox(layoutWidget1);
        checkBox_showFramePointCloud->setObjectName(QString::fromUtf8("checkBox_showFramePointCloud"));
        checkBox_showFramePointCloud->setEnabled(true);
        checkBox_showFramePointCloud->setChecked(true);

        verticalLayout_checkBoxes->addWidget(checkBox_showFramePointCloud);

        checkBox_showTrackletBoundingBoxes = new QCheckBox(layoutWidget1);
        checkBox_showTrackletBoundingBoxes->setObjectName(QString::fromUtf8("checkBox_showTrackletBoundingBoxes"));
        checkBox_showTrackletBoundingBoxes->setChecked(true);

        verticalLayout_checkBoxes->addWidget(checkBox_showTrackletBoundingBoxes);

        checkBox_showTrackletPointClouds = new QCheckBox(layoutWidget1);
        checkBox_showTrackletPointClouds->setObjectName(QString::fromUtf8("checkBox_showTrackletPointClouds"));
        checkBox_showTrackletPointClouds->setChecked(true);

        verticalLayout_checkBoxes->addWidget(checkBox_showTrackletPointClouds);

        checkBox_showTrackletInCenter = new QCheckBox(layoutWidget1);
        checkBox_showTrackletInCenter->setObjectName(QString::fromUtf8("checkBox_showTrackletInCenter"));
        checkBox_showTrackletInCenter->setChecked(true);

        verticalLayout_checkBoxes->addWidget(checkBox_showTrackletInCenter);

        splitterHorizontal->addWidget(layoutWidget1);
        splitterVertical->addWidget(splitterHorizontal);
        qvtkWidget_pclViewer = new QVTKWidget(splitterVertical);
        qvtkWidget_pclViewer->setObjectName(QString::fromUtf8("qvtkWidget_pclViewer"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(qvtkWidget_pclViewer->sizePolicy().hasHeightForWidth());
        qvtkWidget_pclViewer->setSizePolicy(sizePolicy);
        splitterVertical->addWidget(qvtkWidget_pclViewer);

        horizontalLayout_2->addWidget(splitterVertical);

        KittiVisualizerQt->setCentralWidget(centralwidget);
        statusBar = new QStatusBar(KittiVisualizerQt);
        statusBar->setObjectName(QString::fromUtf8("statusBar"));
        KittiVisualizerQt->setStatusBar(statusBar);
        menuBar = new QMenuBar(KittiVisualizerQt);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 989, 25));
        menuFile = new QMenu(menuBar);
        menuFile->setObjectName(QString::fromUtf8("menuFile"));
        KittiVisualizerQt->setMenuBar(menuBar);
        QWidget::setTabOrder(slider_dataSet, slider_frame);
        QWidget::setTabOrder(slider_frame, slider_tracklet);
        QWidget::setTabOrder(slider_tracklet, checkBox_showFramePointCloud);
        QWidget::setTabOrder(checkBox_showFramePointCloud, checkBox_showTrackletBoundingBoxes);
        QWidget::setTabOrder(checkBox_showTrackletBoundingBoxes, checkBox_showTrackletPointClouds);
        QWidget::setTabOrder(checkBox_showTrackletPointClouds, checkBox_showTrackletInCenter);

        menuBar->addAction(menuFile->menuAction());
        menuFile->addAction(actionExit);

        retranslateUi(KittiVisualizerQt);

        QMetaObject::connectSlotsByName(KittiVisualizerQt);
    } // setupUi

    void retranslateUi(QMainWindow *KittiVisualizerQt)
    {
        KittiVisualizerQt->setWindowTitle(QApplication::translate("KittiVisualizerQt", "PCLViewer", 0, QApplication::UnicodeUTF8));
        actionExit->setText(QApplication::translate("KittiVisualizerQt", "Exit", 0, QApplication::UnicodeUTF8));
        actionExit->setShortcut(QApplication::translate("KittiVisualizerQt", "Ctrl+Q", 0, QApplication::UnicodeUTF8));
        label_dataSet->setText(QApplication::translate("KittiVisualizerQt", "Data set:", 0, QApplication::UnicodeUTF8));
        label_frame->setText(QApplication::translate("KittiVisualizerQt", "Frame:", 0, QApplication::UnicodeUTF8));
        label_tracklet->setText(QApplication::translate("KittiVisualizerQt", "Tracklet:", 0, QApplication::UnicodeUTF8));
        checkBox_showFramePointCloud->setText(QApplication::translate("KittiVisualizerQt", "Show frame point cloud", 0, QApplication::UnicodeUTF8));
        checkBox_showTrackletBoundingBoxes->setText(QApplication::translate("KittiVisualizerQt", "Show tracklet bounding boxes", 0, QApplication::UnicodeUTF8));
        checkBox_showTrackletPointClouds->setText(QApplication::translate("KittiVisualizerQt", "Show tracklet point clouds", 0, QApplication::UnicodeUTF8));
        checkBox_showTrackletInCenter->setText(QApplication::translate("KittiVisualizerQt", "Show tracklet in center", 0, QApplication::UnicodeUTF8));
        menuFile->setTitle(QApplication::translate("KittiVisualizerQt", "File", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class KittiVisualizerQt: public Ui_KittiVisualizerQt {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_QTKITTIVISUALIZER_H
