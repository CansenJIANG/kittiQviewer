/****************************************************************************
** Meta object code from reading C++ file 'QtKittiVisualizer.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.6)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../QtKittiVisualizer.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'QtKittiVisualizer.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.6. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_KittiVisualizerQt[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       7,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      25,   19,   18,   18, 0x0a,
      50,   19,   18,   18, 0x0a,
      73,   19,   18,   18, 0x0a,
      99,   19,   18,   18, 0x0a,
     132,   19,   18,   18, 0x0a,
     171,   19,   18,   18, 0x0a,
     208,   19,   18,   18, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_KittiVisualizerQt[] = {
    "KittiVisualizerQt\0\0value\0"
    "newDatasetRequested(int)\0"
    "newFrameRequested(int)\0newTrackletRequested(int)\0"
    "showFramePointCloudToggled(bool)\0"
    "showTrackletBoundingBoxesToggled(bool)\0"
    "showTrackletPointCloudsToggled(bool)\0"
    "showTrackletInCenterToggled(bool)\0"
};

void KittiVisualizerQt::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        KittiVisualizerQt *_t = static_cast<KittiVisualizerQt *>(_o);
        switch (_id) {
        case 0: _t->newDatasetRequested((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 1: _t->newFrameRequested((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 2: _t->newTrackletRequested((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 3: _t->showFramePointCloudToggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 4: _t->showTrackletBoundingBoxesToggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 5: _t->showTrackletPointCloudsToggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 6: _t->showTrackletInCenterToggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData KittiVisualizerQt::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject KittiVisualizerQt::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_KittiVisualizerQt,
      qt_meta_data_KittiVisualizerQt, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &KittiVisualizerQt::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *KittiVisualizerQt::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *KittiVisualizerQt::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_KittiVisualizerQt))
        return static_cast<void*>(const_cast< KittiVisualizerQt*>(this));
    return QMainWindow::qt_metacast(_clname);
}

int KittiVisualizerQt::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 7)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 7;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
