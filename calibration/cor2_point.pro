;+
; $Id: cor2_point.pro,v 1.23 2023/08/15 22:15:08 secchia Exp $
; Project     : STEREO - SECCHI
;
; Name        : COR2_POINT
;
; Purpose     : Correct pointing values in COR2 headers.
;
; Category    : STEREO, SECCHI, COR2, Calibration, Coordinates, attitude, pointing
;
; Explanation : This routine applies the latest pointing calibration to a COR1
;               image header.  The following keywords are affected:
;
;                       CRPIX1, CRPIX2, CRPIX1A, CRPIX2A,
;                       CRVAL1, CRVAL2, CRVAL1A, CRVAL2A,
;                       CDELT1, CDELT2, CDELT1A, CDELT2A,
;                       PC1_1,  PC1_2,  PC2_1,   PC2_2,
;                       PC1_1A, PC1_2A, PC2_1A,  PC2_2A,
;                       CROTA, XCEN, YCEN, ATT_FILE, HISTORY
;
;   	    	USAGE NOTE: Normally, the pointing values in the FITS header include jitter and are accurate 
;   	    	to within 1e-5 arcsec and COR2_POINT SHOULD NOT BE USED! If this pro is called and raw Level-0 
;   	    	($SDS_DIR or $SDS_FIN_DIR) telemetry files are not present, values will be changed to smoothed, 
;   	    	non-jitter case. This is equivalent to using /NOJITTER keyword.
;
; Syntax      : COR2_POINT, HDR
;
; Examples    : A = SCCREADFITS(Filename, HDR)
;               COR2_POINT, HDR
;
; Inputs      : HDR     = SECCHI header structure, or array of structures.  Can
;                         also be passed as a FITS text header, in which case
;                         it will be converted into a SECCHI header structure.
;
; Outputs     : HDR     = The modified header is returned in-place.
;
; Keywords    : Accepts the keywords GTDBFILE, DIR, NO_GRH, and ERROR for the
;               SCC_SUNVEC routine.  See that routine for more details.
;
;   	    	/NOJITTER   Does not use the Level-0 telemetry for deriving jitter (quicker)
;   	    	/NOSPICE    Do not use STEREO SPICE tools to derive s/c roll and coordinate conversion;
;   	    	    	    use header values instead. Note that GEI coordinates are not computed.
;   	    	/ROLLZERO   Assume image has been corrected for spacecraft roll only
;
; Calls       : DATATYPE, SCC_FITSHDR2STRUCT, TAG_NAMES, GET_STEREO_HPC_POINT,
;               GET_STEREO_ROLL, CONVERT_STEREO_LONLAT, FITSHEAD2WCS, SCC_SUNVEC,
;               WCS_GET_COORD, SCC_UPDATE_HISTORY, GET_STEREO_ATT_FILE, GET_ATT_DIR
;
; Written     : K. Baldwin, Mar 2008 (Modified version of cor1_point)
;-
; $Log: cor2_point.pro,v $
; Revision 1.23  2023/08/15 22:15:08  secchia
; roll08/12 correction
;
; Revision 1.22  2016/08/24 22:40:08  thompson
; Corrected post-conjunction calculations.
;
; Revision 1.21  2015/09/30 13:51:42  colaninn
; Updated roll for COR2-B after 2008-06-18 20:10
;
; Revision 1.20  2015/07/14 14:35:23  secchia
; corrected rectify by 180 for post conj
;
; Revision 1.19  2015/07/13 21:19:49  secchia
; removed second rectify correction for crpix
;
; Revision 1.18  2015/07/13 18:57:28  secchia
; added crpix correct for after solar conjuction
;
; Revision 1.17  2015/07/02 12:13:45  mcnutt
; Modified for post-solar-conjunction
;
; Revision 1.16  2014/01/24 18:57:08  nathan
; do not halt if not rectified
;
; Revision 1.15  2013/12/13 19:21:25  colaninn
; changed crpix back; values correct for crval
;
; Revision 1.14  2013/11/27 21:24:11  nathan
; Add 1 to CRPIX values because they were IDL coordinates (not FITS);
; add nominal R1COL/R1ROW to CRPIX so subfield is handled correctly.
;
; Revision 1.13  2013/11/22 19:55:39  nathan
; add in missing parts to account for rectification and subfields
;
; Revision 1.12  2012/09/27 18:41:53  nathan
; add usage note
;
; Revision 1.11  2010/11/17 23:48:03  nathan
; some mods for vector application; more debug msgs
;
; Revision 1.10  2008/10/28 22:01:42  nathan
; check for SPICE_ICY_DLM
;
; Revision 1.9  2008/06/05 14:19:09  baldwin
; CRPIX will remain constant as the occulter center and CRVAL will correct for the pointing offset.
;
; Revision 1.8  2008/05/02 20:36:25  nathan
; added /ROLLZERO
;
; Revision 1.7  2008/05/02 19:52:08  nathan
; added /NOJITTER and /NOSPICE options
;
; Revision 1.6  2008/05/02 16:01:45  baldwin
; No change.
;
; Revision 1.4  2008/04/03 18:19:23  nathan
; print file version and add to history
;
; Revision 1.3  2008/03/27 19:50:15  nathan
; added cvs tags, updated headers
;
; --------------Hardcoded telemdir and removed rectify related lines--------------
;
pro cor2_point, hdr, gtdbfile=gtdbfile, dir=tdir, no_grh=no_grh, NOJITTER=nojitter, $
                error=error, SILENT=silent, NOSPICE=nospice, ROLLZERO=rollzero, _EXTRA=_extra

info="$Id: cor2_point.pro,v 1.23 2023/08/15 22:15:08 secchia Exp $"
loud=~keyword_set(SILENT)
IF loud THEN print,info
;
;  Make sure that the header is a SECCHI structure, and that it belongs to
;  COR2.
;
if datatype(hdr) ne 'STC' then hdr=scc_fitshdr2struct(hdr)
if tag_names(hdr, /structure_name) ne 'SECCHI_HDR_STRUCT' then message, $
  'Only SECCHI header structure allowed'
w = where(hdr.detector ne 'COR2', count)
if count gt 0 then message, 'Calibration for COR2 detector only'
;
;  Take into account the image rotation, if any.
;
wr = where(hdr.rectify NE 'T', countr)
if countr gt 0 then BEGIN
    message,'Sorry! COR2_POINT.PRO only works on rectified images.',/info
    message,'Returning hdr unchanged.',/info
    wait,5
    return
ENDIF
;
;  Define the default parameters based on the observatory.
;
nhdr   = n_elements(hdr)
crpix  = fltarr(2,nhdr)         ;Reference pixel
crval  = fltarr(2,nhdr)         ;Reference value
cdelt  = fltarr(nhdr)           ;Plate scale
spoint = fltarr(3,nhdr)         ;Solar pointing (yaw, pitch, roll)
drot  = fltarr(nhdr)            ;Rotation correction
croll  = fltarr(nhdr)           ;Celestial roll
cpoint = fltarr(2,nhdr)         ;Celestial yaw and pitch
wa = where(hdr.obsrvtry eq 'STEREO_A', na, complement=wb, ncomplement=nb)
pcdate0 = '2015-05-19'          ;Start of post-conjunction
pcdate1 = '2023-08-12'          ;End of post-conjunction
;
;  Define the defaults for Ahead.  The pitch and yaw parts of GET_STEREO_ROLL
;  and GET_STEREO_HPC_POINT are ignored in favor of the Guide Telescope data.
;
if na gt 0 then begin
    ;These crpix do not match output of getscccrpix.pro
    ;These values are correct for crval below!!!!
    crpix[0,wa] = 1022.0 ;near occulter center
    crpix[1,wa] = 1020.8 ;-1 measured pre-launch
    ;Correct of over/under scan r1row/col 
    crpix[0,wa] = crpix[0,wa] + 129
    crpix[1,wa] = crpix[1,wa] + 51
    ;HPC Coordinate values of CRPIX
    crval[0,wa] =  67.4553 
    crval[1,wa] = -55.5457	
    ;Platescale arcseconds 
    cdelt[wa] = 14.7
    ;Instrument roll offset degrees
    drot[wa]   = 0.45
    date_obs = hdr[wa].date_obs

;restore,'./cor2point2009.sav'
;goto, skiptemp
    IF keyword_set(NOSPICE) or ~TEST_SPICE_ICY_DLM() THEN BEGIN
    	spoint[2,wa] = hdr[wa].sc_roll 
	FOR i=0,na-1 DO BEGIN
	    wcsa=fitshead2wcs(hdr[wa[i]],system='a')
	    croll[wa[i]]=wcsa.roll_angle
	ENDFOR
	att_file = hdr[wa].att_file
	dsun = hdr[wa].dsun_obs*1e-3/oneau('km')
    ENDIF ELSE BEGIN
	IF loud and na GT 0 THEN print,'Running get_stereo_hpc_point...'
    	spoint[2,wa] = (get_stereo_hpc_point(date_obs, 'A', /degrees,/post_conjunction))[2,*]
	IF loud and na GT 0 THEN print,'Running get_stereo_roll...'
	croll[wa] = get_stereo_roll(date_obs, 'A', system='GEI', /degrees,/post_conjunction)
	IF loud and na GT 0 THEN print,'Running get_stereo_att_file...'
	att_file = get_stereo_att_file(date_obs,'A')  ; much faster
	IF loud and na GT 0 THEN print,'Running get_stereo_lonlat...'
    	dsun = (get_stereo_lonlat(hdr[wa].date_obs, 'a', /au))[0,*]
    ENDELSE
;
;  Fold in the Guide Telescope data.
;
    date0 = '2007-07-01T01:07:30.006'	;= 20070701_010730_d4c2A.fts
	;date0 = '2007-09-03T18:41:37.847'
    ;dsun0 = (get_stereo_lonlat(date0, 'a', /au))[0]
    dsun0 = 0.95688249
    IF loud and na GT 1 THEN print,'Running scc_sunvec...'
    d0 = scc_sunvec(date0, dsun0, obs='a', /quiet, gtdbfile=gtdbfile)
    for ia=0,na-1 do begin
        ja = wa[ia]
	IF keyword_set(NOJITTER) THEN telemdir='' ELSE $
	IF keyword_set(DIR) THEN telemdir=tdir ELSE telemdir = get_att_dir(hdr[ja].date_obs, 'a')
        d =scc_sunvec(hdr[ja].date_obs, dsun[ia], obs='a', gtdbfile=gtdbfile, $
                      dir=telemdir, no_grh=no_grh, error=error, quiet=silent)
        if (hdr[ja].date_obs ge pcdate0) and (hdr[ja].date_obs lt pcdate1) $
                then sgn = 1 else sgn = -1
         spoint[0,ja] = sgn * (d[1]-d0[1]) / 3600.d0
         spoint[1,ja] = sgn * (d[0]-d0[0]) / 3600.d0
    endfor
;
    point = spoint[0:1,wa]
    IF loud and na GT 1 THEN print,'Running convert_stereo_lonlat...'
    IF keyword_set(NOSPICE) THEN BEGIN
    	IF ~keyword_set(SILENT) THEN message,'corrected GEI coordinates not derived',/info
    ENDIF ELSE $
    convert_stereo_lonlat, date_obs, point, 'HPC', 'GEI', spacecraft='A', $
      /degrees
    cpoint[*,wa] = point
;skiptemp:
endif
;
;  Define the defaults for Behind.
;
if nb gt 0 then begin
    ;These crpix do not match output of getscccrpix.pro
    ;These values are correct for crval below!!!!
    crpix[0,wb] = 1020.60 ;near occulter center
    crpix[1,wb] = 1033.20 ;-1 measured pre launch
    ;Correct of over/under scan r1row/col 
    crpix[0,wb] = crpix[0,wb] + 1
    crpix[1,wb] = crpix[1,wb] + 79
    ;HPC Coordinate values of CRPIX
    crval[0,wb] = -130.938087 
    crval[1,wb] = 115.425487
    ;Platescale arcseconds
    cdelt[wb] = 14.7
    ;Instrument roll offset degrees
    drot[wb]  = -0.20
    wr = where(anytim2tai(hdr[wb].date_obs) gt anytim2tai('2008-06-18 20:10'),nr)
    IF nr GT 0 THEN drot[wb[wr]] = -0.30

    date_obs = hdr[wb].date_obs
    IF keyword_set(NOSPICE) or ~TEST_SPICE_ICY_DLM() THEN BEGIN
    	spoint[2,wb] = hdr[wb].sc_roll 
	FOR i=0,nb-1 DO BEGIN
	    wcsa=fitshead2wcs(hdr[wb[i]],system='a')
	    croll[wb[i]]=wcsa.roll_angle
	ENDFOR
	att_file = hdr[wb].att_file
	dsun = hdr[wb].dsun_obs*1e-3/oneau('km')
    ENDIF ELSE BEGIN
	IF loud and nb GT 0 THEN print,'Running get_stereo_hpc_point...'
	spoint[2,wb] = (get_stereo_hpc_point(date_obs, 'B', /degrees,/post_conjunction))[2,*]
	IF loud and nb GT 0 THEN print,'Running get_stereo_roll...'
	croll[wb] = get_stereo_roll(date_obs, 'B', system='GEI', /degrees,/post_conjunction)
	IF loud and nb GT 0 THEN print,'Running get_stereo_att_file...'
	att_file = get_stereo_att_file(date_obs,'B')  ; much faster
	IF loud and nb GT 0 THEN print,'Running get_stereo_lonlat...'
    	dsun = (get_stereo_lonlat(hdr[wb].date_obs, 'b', /au))[0,*]
    ENDELSE
;
;  Fold in the Guide Telescope data.
;
    date0 = '2007-05-01T01:08:11.422'
    ;dsun0 = (get_stereo_lonlat(date0, 'b', /au))[0]
    dsun0 = 1.0442248
    IF loud and nb GT 1 THEN print,'Running scc_sunvec...'
    d0 = scc_sunvec(date0, dsun0, obs='b', /quiet, gtdbfile=gtdbfile)
    for ib=0,nb-1 do begin
        jb = wb[ib]
	IF keyword_set(NOJITTER) THEN telemdir='' ELSE $
	IF keyword_set(DIR) THEN telemdir=tdir ELSE telemdir = get_att_dir(hdr[jb].date_obs, 'b')
    	d =scc_sunvec(hdr[jb].date_obs, dsun[ib], obs='b', gtdbfile=gtdbfile, $
                      dir=telemdir, no_grh=no_grh, error=error, quiet=silent)
        if (hdr[jb].date_obs ge pcdate0) and (hdr[jb].date_obs lt pcdate1) $
                then sgn = -1 else sgn = 1
         spoint[0,jb] = sgn * (d[1]-d0[1]) / 3600.d0
         spoint[1,jb] = sgn * (d[0]-d0[0]) / 3600.d0 
    endfor
;
    point = spoint[0:1,wb]
    IF loud and nb GT 1 THEN print,'Running convert_stereo_lonlat...'
    IF keyword_set(NOSPICE) or ~TEST_SPICE_ICY_DLM() THEN BEGIN
    	IF ~keyword_set(SILENT) THEN message,'corrected GEI coordinates not derived',/info
    ENDIF ELSE $
    convert_stereo_lonlat, date_obs, point, 'HPC', 'GEI', spacecraft='B', $
      /degrees
    cpoint[*,wb] = point
endif
IF keyword_set(ROLLZERO) THEN BEGIN
    spoint[2,*]=spoint[2,*]-hdr.sc_roll
ENDIF
    
; Adjust any post-conjunction data.
;
wdate = where((hdr.date_obs ge pcdate0) and (hdr.date_obs lt pcdate1), ndate)
if ndate gt 0 then begin
    crval[*,wdate] = -crval[*,wdate]
    wrect = wdate[where(hdr[wdate].rectify eq 'T', nrect)]
    if nrect gt 0 then crpix[*,wrect] = 2179 - crpix[*,wrect]
endif
;
; Set instrument offset keywords
;
hdr.ins_x0 = reform(crval[0,*]) ; arcsec
hdr.ins_y0 = reform(crval[1,*]) ; arcsec
hdr.ins_r0 = drot       ; deg
;
;  Set SC_ values with GT values before transform
;
hdr.sc_yaw = reform(spoint[0,*])
hdr.sc_pitch=reform(spoint[1,*])
hdr.sc_yawa = reform(cpoint[0,*])
hdr.sc_pita = reform(cpoint[1,*])

;
;  Take into account the area of the CCD used.
;

crpix[0,*] = crpix[0,*] - hdr.r1col
crpix[1,*] = crpix[1,*] - hdr.r1row
;
;
;  Correct the plate scale and reference pixels for pixel summing.
;
summed = 2.^(hdr.summed - 1)    ;Amount of image summing
w = where(summed gt 1, count)
if count gt 0 then begin
    cdelt[w] = cdelt[w] * summed[w]
    crpix[0,w] = (crpix[0,w] - 0.5) / summed[w] + 0.5
    crpix[1,w] = (crpix[1,w] - 0.5) / summed[w] + 0.5
endif
;
;  Step through the dates and transform the coordinates based on the spacecraft
;  pointing information.
;
dr = !dpi / 180.d0              ;Degrees to radians
ddr = dr / 3600.d0              ;Arcseconds to radians
;
for ihdr = 0,nhdr-1 do begin
;
;  Form the transformation matrix from the sines and cosines of the spacecraft
;  sun-pointing parameters.
;
    sy = sin(spoint[0,ihdr]*dr)  &  cy = cos(spoint[0,ihdr]*dr)
    sp = sin(spoint[1,ihdr]*dr)  &  cp = cos(spoint[1,ihdr]*dr)
    sr = sin(spoint[2,ihdr]*dr)  &  cr = cos(spoint[2,ihdr]*dr)
;
    mat = [[cp*cy, cp*sy, -sp], $
           [-cr*sy+sr*sp*cy,  cr*cy+sr*sp*sy, sr*cp], $
           [ sr*sy+cr*sp*cy, -sr*cy+cr*sp*sy, cr*cp]]
;
;  Form the vector representing the pointing of the instrument.
;
    yaw   = -crval[0,ihdr] * ddr
    pitch =  crval[1,ihdr] * ddr
    vec = [cos(yaw)*cos(pitch), sin(yaw)*cos(pitch), sin(pitch)]
;
;  Transform the pointing and recalculate the pointing.
;
    newvec = mat ## vec
    crval[0,ihdr] = -atan(newvec[1], newvec[0]) / ddr
    crval[1,ihdr] =  asin(newvec[2])            / ddr
;
;  Convert the pointing into celestial coordinates.
;
    point = crval[0:1,ihdr] / 3600.d0
    IF ~keyword_set(NOSPICE) and TEST_SPICE_ICY_DLM() $
    THEN convert_stereo_lonlat, date_obs[ihdr], point, 'HPC', 'GEI', /degrees, spacecraft=hdr[ihdr].obsrvtry
    cpoint[0:1,ihdr] = point
endfor
;

hdr.crpix1  = reform(crpix[0,*])
hdr.crpix1a = hdr.crpix1
hdr.crpix2  = reform(crpix[1,*])
hdr.crpix2a = hdr.crpix2
;
hdr.cdelt1  = cdelt
hdr.cdelt2  = cdelt
ccdelt = cdelt / 3600.0         ;Convert arcseconds to degrees
hdr.cdelt1a = -ccdelt           ;Negative for celestial coordinates
hdr.cdelt2a =  ccdelt
;
hdr.crval1  = reform(crval[0,*])
hdr.crval2  = reform(crval[1,*])
hdr.crval1a = reform(cpoint[0,*])
hdr.crval2a = reform(cpoint[1,*])
;
;  Add either 90 or 270 degrees to the rotation angle for unrectified images.
;
gamma = reform(spoint[2,*]) + drot
wa = where((hdr.rectify ne 'T') and (hdr.obsrvtry eq 'STEREO_A'), counta)
if counta gt 0 then gamma[wa] = gamma[wa] + 270 
wb = where((hdr.rectify ne 'T') and (hdr.obsrvtry eq 'STEREO_B'), countb)
if countb gt 0 then gamma[wb] = gamma[wb] + 90 
;
;  Put the correct rotation angle and matrices in the headers.
;
hdr.crota = gamma
gamma = gamma * dr              ;Convert degrees to radians
cgamma = cos(gamma)
sgamma = sin(gamma)
hdr.pc1_1 =  cgamma
hdr.pc1_2 = -sgamma
hdr.pc2_1 =  sgamma
hdr.pc2_2 =  cgamma
;
;  Calculate XCEN, YCEN
;
di = (hdr.naxis1 + 1.)/2. - reform(crpix[0,*])
dj = (hdr.naxis2 + 1.)/2. - reform(crpix[1,*])
hdr.xcen = reform(crval[0,*]) + cdelt*(di*cgamma - dj*sgamma)
hdr.ycen = reform(crval[1,*]) + cdelt*(di*sgamma + dj*cgamma)
;
;  Do the same for celestial coordinates.
;
gamma = (croll - drot) * dr
cgamma = cos(gamma)
sgamma = sin(gamma)
hdr.pc1_1a =  cgamma
hdr.pc1_2a =  sgamma
hdr.pc2_1a = -sgamma
hdr.pc2_2a =  cgamma

IF error GT 0 THEN errstr=trim(fix(error)) ELSE errstr=''
att_file = att_file+'+'+errstr+'GT'
; update att_file
if tag_exist(hdr,'att_file') then hdr.att_file = att_file
if ~keyword_set(silent) then help,error,att_file

len=strlen(info)
histinfo=strmid(info,1,len-2)
IF na+nb Gt 1 THEN print,'Not updating HISTORY.' ELSE $
if tag_exist(hdr,'history') then hdr=scc_update_history(hdr, histinfo)

return
end

