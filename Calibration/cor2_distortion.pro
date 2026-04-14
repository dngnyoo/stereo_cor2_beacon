;+
; $Id: cor2_distortion.pro,v 1.14 2011/07/22 16:22:11 avourlid Exp $
; NAME: COR2_DISTORTION
;
; PURPOSE:
;       This function returns distance in pixels from CRPIX after distortion
;       correction has been applied.
;
; CATEGORY: calibration distortion coordinates 
;
; CALLING SEQUENCE:
;       RESULT = COR2_DISTORTION(data)
;
; INPUTS:
;       Data - number scalar or array - distance(s) in pixels from CRPIX
;       Spacecraft - STRING - indicating spacecraft A or B
;       secs - FLOAT - Platescale in arcsec (hdr.cdelt1)
;
; OUTPUT:
;       Result:      new, undistorted distance(s) in pixels
;
; OPTIONAL OUTPUTS:
;       COEF=   Named variable contains coefficients used in correction
;       INFO=   Named variable contains CVS revision info
;
; PROCEDURE:
;       Given the distance between CRPIX1,CRPIX2 and a point (x, y)
;       on the CCD, in pixels, this function returns the distance to
;       (x, y) in pixels after a function application.
;
;  Written:    K. Baldwin 11 Mar 2008 
;-
; $Log: cor2_distortion.pro,v $
; Revision 1.14  2011/07/22 16:22:11  avourlid
; Removed plate scale adjustment
;
; Revision 1.13  2009/02/12 21:55:19  nathan
; replaced data with data0, as in rev 1.10
;
; Revision 1.12  2009/02/12 21:33:48  baldwin
; Re-Added plate scale argument.
;
; Revision 1.10  2008/08/12 19:10:54  nathan
; removed extraneous code that was introduced when a conflict was committed into repository in rev 1.7
;
; Revision 1.9  2008/08/06 20:05:58  baldwin
; Slight adjustments made to equation coefficients.
;
; Revision 1.8  2008/06/16 16:14:37  reduce
; Another bug fix. 'sc' was not defined. KB
;
; Revision 1.7  2008/06/16 16:07:52  reduce
; Commented out stray <<<'s, ==='s, and >>>'s. KB
;
; Revision 1.6  2008/05/02 15:09:17  nathan
; changed behavior of platescale input: normalizes to full res before applying function
;
; Revision 1.5  2008/04/29 19:36:17  baldwin
; Added plate scale argument
;
; Revision 1.4  2008/04/29 18:34:04  nathan
; Use parse_stereo_name.pro; convert input to type FLOAT; add keyword outputs
; COEF and INFO; update comments
;
; Revision 1.3  2008/03/27 19:50:15  nathan
; added cvs tags, updated headers
;
FUNCTION cor2_distortion, data, spacecraft, secs, COEF=cf, INFO=info, _EXTRA=_extra

; ==================================================================================

sc=parse_stereo_name(spacecraft,['A','B'])
; secs0=getsccsecpix('cor2',sc,/silent) ;not necessary for binned images-AV

if (sc EQ 'A') then $
    cf = [1.04872e-05, -0.00801293, -0.243670] ELSE $  ; COR2A coeffs (pixels)
    cf = [1.96029e-05, -0.0201616,   4.68841]          ; COR2B coeffs (pixels)
    
data0=float(data) ;*secs/secs0 ;not necessary for binned images-AV
f1 = data0 + (cf[2]+(cf[1]*data0)+(cf[0]*(data0*data0)))    ; apply distortion function

id='$Id: cor2_distortion.pro,v 1.14 2011/07/22 16:22:11 avourlid Exp $'
len=strlen(id)
info=strmid(id,1,len-2)
data0=0
return, f1  ;* secs0/secs ;not necessary for binned images-AV
end

