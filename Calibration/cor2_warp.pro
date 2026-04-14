;+
; $Id: cor2_warp.pro,v 1.15 2015/09/30 13:51:42 colaninn Exp $
;
; NAME:     COR2_Warp
;
; PURPOSE: 
;
; CATEGORY: calibration distortion coordinates 
;
; CALLING SEQUENCE:
;       RESULT = COR2_WARP(image, header)
;
; INPUTS:  secchi cor2a or cor2b image and fits header
;
; OUTPUT:   warped image of original secchi cor2a or cor2b filename -
;               corrects for distortion. Calibration based on 2048 x 2048 images.
;		Subfield or binned images are not handled properly. 
;
; OPTIONAL OUTPUTS:
;   	INFO=	Named variable contains 2-elem STRARR with version info  
;
; Calls : cor2_distortion, warp_tri     
;   
; Written:    K. Baldwin 11 Mar 2008 (Modified version of C2_warp)
;-
; $Log: cor2_warp.pro,v $
; Revision 1.15  2015/09/30 13:51:42  colaninn
; Updated roll for COR2-B after 2008-06-18 20:10
;
; Revision 1.14  2013/03/21 20:31:34  nathan
; change DISTCORR in input hdr, print message, and implement /SILENT
;
; Revision 1.13  2009/11/17 15:18:58  mcnutt
; adjust scalef for summed images
;
; Revision 1.12  2009/11/17 14:28:57  mcnutt
; will now work with mvi frame headers
;
; Revision 1.11  2009/02/17 22:43:20  baldwin
; sumxy = 2^(hdr.ipsum + hdr.ccdsum -2) -> sumxy = 2^(hdr.summed-1)
; scalef = hdr.cdelt1 -> scalef = 14.7
;
; remove ->  scalef = scalef * (hdr.CCDSUM > 1)  remove ->  scalef = scalef * (hdr.IPSUM > 1)
;
; Revision 1.10  2009-02-12 21:38:17  baldwin
; Reapplied 'scalef' variable to make program compatible with cor2_distortion.
;
; Revision 1.8  2008/06/16 18:50:34  mcnutt
; does not correct sun center for summing already corrected
;
; Revision 1.7  2008/06/04 19:11:50  baldwin
; Removed extra scalef multiplication.
;
; Revision 1.6  2008/05/02 19:59:36  nathan
; added optional output INFO=
;
; Revision 1.5  2008/05/02 15:59:41  baldwin
; Sun's center is used as distortion axis.
;
; Revision 1.4  2008/04/29 19:35:07  baldwin
; Now plate scale is taken into account.
;
; Revision 1.3  2008/03/27 19:50:15  nathan
; added cvs tags, updated headers
;
function cor2_warp, image, hdr, INFO=histinfo, SILENT=silent, _EXTRA=_extra

; ==================================================================================
; Establish control points x and y at every 32 pixel:
  gridsize=32L
  w = indgen(((2048L/gridsize)+1L)^2L)
  y = w/((2048L/gridsize)+1L)
  x = w-y*((2048L/gridsize)+1L)
  x = x*gridsize & y= y*gridsize
; ==================================================================================
; ==================================================================================
if tag_exist(hdr,'OBSRVTRY') then begin
; determine key variables
  wcs=fitshead2wcs(hdr,system='A')
  sc = hdr.OBSRVTRY
  sc = strcompress(sc,/remove_all)
  sc = strmid(sc,strlen(sc)-1,1)
; Find corresponding control points x0 and y0 from x and y: 
; Warp is determined as a function of sun's center.
  
; Finds sun center in pixels.  Note that scc_sun_center returns the
; sun center based on information in the image header; therefore, this
; value has been scaled based on the image size.  In other words, for
; beacom data, suncen=[128,128] (approximately)
  suncen = scc_sun_center(hdr)
  xc = suncen.xcen
  yc = suncen.ycen

; Calculate the amount of binning that has been applied to the input
; image; a similar amount of binning must shortly be applied to the
; control points.

  sumxy = 2^(hdr.summed-1) 

endif else begin
; ==================================================================================
; get values from mvi header
   sc = strmid(hdr.filename,strpos(hdr.filename,'.ft')-1,1)
   xc= hdr.xcen
   yc= hdr.ycen
   sumxy= hdr.cdelt1/14.7    ;arcsec/pixel for full res COR2 A and B
endelse

; Commented out scalef since it is not needed; all warping
; calculations are done in pixels
; scalef = hdr.cdelt1
 scalef = 14.7 *sumxy
; scalef = scalef * (hdr.CCDSUM > 1)     ;apply correction for on chip summing
; scalef = scalef * (hdr.IPSUM > 1)      ;apply correction for SEB summing

; Note that x and y are based on a full-sized image of 2048x2048, but
; xc and yc are relative to a possibly binned image; hence, we
; multiply xc and yc by sumxy
  r= sqrt((x-sumxy*xc)^2+(y-sumxy*yc)^2)

; Adjust r0, x, y to input image size
  r0 = cor2_distortion(r, sc, scalef,INFO=histinfo) / sumxy
  x  = x / sumxy
  y  = y / sumxy

  id='$Id: cor2_warp.pro,v 1.15 2015/09/30 13:51:42 colaninn Exp $'
  len=strlen(id)
  histinfo=[histinfo,strmid(id,1,len-2)]

; Find the control points corresponding to x and y
  theta= atan((y-yc),(x-xc))
  x0= r0*cos(theta)+xc
  y0= r0*sin(theta)+yc
; ============================================================================

  hdr.distcorr='T'
  IF ~keyword_set(SILENT) THEN message,'Distortion correction applied.',/info
; Distort the image by shifting locations (x,y) to (x0,y0) and return it:
  return, warp_tri(x,y,x0,y0,image)  
end

