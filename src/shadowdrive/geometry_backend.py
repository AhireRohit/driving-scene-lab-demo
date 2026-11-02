"""Batch the upstream GIoU equation with GEOS; no association policy changes."""
import numpy as np
import shapely


def giou_distance(dets, tracks):
    shape=(len(dets),len(tracks))
    if not dets or not tracks:
        return np.empty(shape)
    # Upstream bottom corners are a rectangle centered on (x,y), yaw about z.
    def data(boxes):
        x=np.array([b.x for b in boxes]);y=np.array([b.y for b in boxes])
        l=np.array([b.l for b in boxes]);w=np.array([b.w for b in boxes])
        h=np.array([b.h for b in boxes]);z=np.array([b.z for b in boxes])
        yaw=np.array([b.o for b in boxes]);c=np.cos(yaw);s=np.sin(yaw)
        dx=np.array([1,1,-1,-1])[None,:]*l[:,None]/2
        dy=np.array([-1,1,1,-1])[None,:]*w[:,None]/2
        corners=np.stack([x[:,None]+c[:,None]*dx-s[:,None]*dy,
                          y[:,None]+s[:,None]*dx+c[:,None]*dy],axis=-1)
        return corners,l*w*h,z,h
    a,va,za,ha=data(dets);b,vb,zb,hb=data(tracks)
    pa=shapely.polygons(a);pb=shapely.polygons(b)
    overlap=shapely.area(shapely.intersection(pa[:,None],pb[None,:]))
    height1=(za+ha/2)[:,None]-(zb-hb/2)[None,:]
    height2=(zb+hb/2)[None,:]-(za-ha/2)[:,None]
    overlap_height=np.maximum(0,np.minimum(np.minimum(height1,height2),np.minimum(ha[:,None],hb[None,:])))
    union_height=np.maximum(np.maximum(height1,height2),np.maximum(ha[:,None],hb[None,:]))
    intersection=overlap*overlap_height
    union=va[:,None]+vb[None,:]-intersection
    allcorners=np.concatenate([np.broadcast_to(a[:,None],(*shape,4,2)),np.broadcast_to(b[None,:],(*shape,4,2))],axis=2)
    hull_area=shapely.area(shapely.convex_hull(shapely.multipoints(allcorners)))
    enclosing=hull_area*union_height
    return 1-(intersection/union-(enclosing-union)/enclosing)
