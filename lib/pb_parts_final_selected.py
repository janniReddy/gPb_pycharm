import cv2
import math
import numpy as np

from lib_image import *

def pb_parts_final_selected(L, a, b):
    print ("inside pb_parts_final_selected function")
    # parameters - binning and smoothing
    n_ori = 8                              # number of orientations 
    num_L_bins = 25                        # # bins for bg 
    num_a_bins = 25                        # # bins for cg_a 
    num_b_bins = 25                        # # bins for cg_b 
    bg_smooth_sigma = 0.1                  # bg histogram smoothing sigma 
    cg_smooth_sigma = 0.05                 # cg histogram smoothing sigma 
    border = 30                            # border pixels 
    # sigma_tg_filt_sm = 2.0                  # sigma for small tg filters 
    # sigma_tg_filt_lg = math.sqrt(2) * 2.0   # sigma for large tg filters 

    # parameters - radii
    n_bg = 3
    n_cg = 3
    n_tg = 3
    r_bg = [ 3, 5, 10 ]
    r_cg = [ 5, 10, 20 ]
    r_tg = [ 5, 10, 20 ]

    # convert to grayscale
    g_im = grayscale(L, a, b)

    # mirror border
    L = cv2.copyMakeBorder(L, border, border, border, border, cv2.BORDER_REFLECT)
    a = cv2.copyMakeBorder(a, border, border, border, border, cv2.BORDER_REFLECT)
    b = cv2.copyMakeBorder(b, border, border, border, border, cv2.BORDER_REFLECT)
    
    # gamma correct
    np.power(L, 2.5)
    np.power(a, 2.5)
    np.power(b, 2.5)
    # rgb_im_temp = cv2.merge([L,a,b])
    # print (rgb_im_temp.shape)

    # Convert to Lab color space
    L, a, b = rgb_to_lab(L, a, b)
    L, a, b = lab_normalize(L, a, b)

    # quantize color channels
    Lq = quantize_values(L, num_L_bins)
    aq = quantize_values(a, num_a_bins)
    bq = quantize_values(b, num_b_bins)

    # compute texton filter set
    # filters_large = texton_filters(n_ori, sigma_tg_filt_lg)
    print("before filters")
    filters = texton_filters(n_ori)
    print("after filters")
    print(len(filters))
    
    k = 64

    # compute textons
    print("before compute_textons called")
    print(g_im.shape)
    textons, t_assign = compute_textons(g_im, border, filters, k)
    print("after compute_textons called")
    print(textons.shape)

    # # writing textons object to a file
    # fo = open("testing_map_obj2.txt", "w")
    # for i in range(0, textons.shape[0]):
    #     for j in range(0, textons.shape[1]):
    #         fo.write(textons[i][j].astype(str))
    #         if j != textons.shape[1] - 1:
    #             fo.write(",")
    #     fo.write("\n")
    # fo.close()
    # print("after writing the textons object")

    # compute bg histogram smoothing kernel
    bg_smooth_kernel = gaussian(sigma = bg_smooth_sigma*num_L_bins)
    cga_smooth_kernel = gaussian(sigma = cg_smooth_sigma*num_a_bins)
    cgb_smooth_kernel = gaussian(sigma = cg_smooth_sigma*num_b_bins)

    # compute bg at each radius
    print("computing bg's")
    bg_r3 = hist_gradient_2D(Lq, r_bg[0], n_ori, bg_smooth_kernel)
    bg_r5 = hist_gradient_2D(Lq, r_bg[1], n_ori, bg_smooth_kernel)
    bg_r10 = hist_gradient_2D(Lq, r_bg[2], n_ori, bg_smooth_kernel)
    print ("after bg_r3 computing")
    print (len(bg_r3))
    print(bg_r3[0].shape)
    for n in range(0, n_ori):
        bg_r3[n] = border_trim_2D(bg_r3[n], border)
        bg_r5[n] = border_trim_2D(bg_r5[n], border)
        bg_r10[n] = border_trim_2D(bg_r10[n], border)
    
    # compute cga at each radius
    print("computing cga's")
    cga_r5 = hist_gradient_2D(aq, r_cg[0], n_ori, cga_smooth_kernel)
    cga_r10 = hist_gradient_2D(aq, r_cg[1], n_ori, cga_smooth_kernel)
    cga_r20 = hist_gradient_2D(aq, r_cg[2], n_ori, cga_smooth_kernel)
    for n in range(0, n_ori):
        cga_r5[n] = border_trim_2D(cga_r5[n], border)
        cga_r10[n] = border_trim_2D(cga_r10[n], border)
        cga_r20[n] = border_trim_2D(cga_r20[n], border)

    # compute cgb at each radius 
    print("computing cgb's")
    cgb_r5 = hist_gradient_2D(bq, r_cg[0], n_ori, cgb_smooth_kernel)
    cgb_r10 = hist_gradient_2D(bq, r_cg[1], n_ori, cgb_smooth_kernel)
    cgb_r20 = hist_gradient_2D(bq, r_cg[2], n_ori, cgb_smooth_kernel)
    for n in range(0, n_ori):
        cgb_r5[n] = border_trim_2D(cgb_r5[n], border)
        cgb_r10[n] = border_trim_2D(cgb_r10[n], border)
        cgb_r20[n] = border_trim_2D(cgb_r20[n], border)
    
    # compute tg at each radius
    np_empty = np.array([])
    print("computing tg's")
    tg_r5 = hist_gradient_2D(t_assign, r_tg[0], n_ori, np_empty)
    tg_r10 = hist_gradient_2D(t_assign, r_tg[1], n_ori, np_empty)
    tg_r20 = hist_gradient_2D(t_assign, r_tg[2], n_ori, np_empty)
    for n in range(0, n_ori):
        tg_r5[n] = border_trim_2D(tg_r5[n], border)
        tg_r10[n] = border_trim_2D(tg_r10[n], border)
        tg_r20[n] = border_trim_2D(tg_r20[n], border)

    return [textons, bg_r3, bg_r5,  bg_r10,  cga_r5, cga_r10, cga_r20, cgb_r5, cgb_r10, cgb_r20, tg_r5,  tg_r10,  tg_r20]