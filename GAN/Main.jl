using PyCall

include("GAN_lenses.jl")
include("helpers.jl")

# Load images, Normalise_images, and create the train / test sets
load_length = 606
start_point = 0

# train_images  = load_Background(load_length, start_point)
# Normalise_images(train_images)

train_images  = load_lenses(load_length, start_point, "GAN_lenses_generation/DES-candidates/")
Normalise_images(train_images)

#Augment the images by rotating them 3 times.
train_images = augment_rot(train_images)


# train_images  = load_lenses_CFIS(load_length, start_point, "spirals")
# Normalise_images_CFIS(train_images)


train_images .*= 250
train_labels = ones(load_length*4) # Dummy labels, could then create multiple classes maybe.


# train_images, train_labels , test_images , test_labels = create_training_test(background_img,Lenses_img, 0.9)

# py"test_train_julia"(train_images , train_labels)
# images_sample, loss =  py"train_pixel_cnn"(train_images , train_labels, train_images, 7000, true)
images_sample, loss =  py"train_pixel_cnn"(train_images , train_labels, train_images, 1000, true)


using Plots
pgfplotsx() # for detailed latex plots
theme(:wong)
default(titlefont = (20, "times"), legendfontsize = 10, guidefont = (16, :black),
    tickfont = (12, :black), framestyle = :box,legend = :topright, markerstrokewidth = 0.3,
    minorgrid = true, minorgridlinewidth= 0.6, gridalpha = 0.38 , minorticks = true ,
    gridlinewidth = 1.2  ,  dpi = 300)

plot(1:length(loss), abs.(loss).+0.0001 , lw = 2, legend = false)
xaxis!("Epochs")
yaxis!("Loss")
savefig("Learn_lenses_final.pdf")




# Generated and diplay + save images ==================================================================

images_sample = py"predict_pixel_cnn"(101, 100 , "PixelCNN_long_final.h5")

to_save = []
for k = 1:length(images_sample[:,1,1,1,1])
     global to_save = vcat(to_save, images_sample[k,:,:,:,:])
end
# to save the image sample.
Normalise_images(to_save)
save_generated_images(to_save)

# plot the image samples
using Plots
using Images , ImageView
using MosaicViews, ImageShow
theme(:wong2)
pyplot()

images_sample = float32.(to_save)
Normalise_images(images_sample)

Final_image = [colorview(RGB , permutedims( images_sample[i,:,:,:] , [3,1,2])) for i = 1:length(images_sample[:,1,1,1])]

# Dsiplay mosaic of images.
img = mosaicview(Final_image,fillvalue=.95, nrow=10, npad=1)
Images.save("Mosaic_test_lense_final_100.png",img)


# Section to check the magnitude histograms =============================================================

# load the registered generated images.
images_sample  = load_lenses_Gen(load_length, start_point)
Normalise_images(images_sample)

centered_gen_img = circular_cut(images_sample[1:606,:,:,:] , 21)
centered_gen_hist = compute_mag_hist_RGB(centered_gen_img)

centered_real_img = circular_cut(train_images , 21)
centered_real_hist = compute_mag_hist_RGB(centered_real_img)

using Plots
pgfplotsx() # for detailed latex plots
theme(:wong)
default(titlefont = (14, "times"), legendfontsize = 8, guidefont = (12, :black),
    tickfont = (10, :black), framestyle = :box,legend = :topright, markerstrokewidth = 0.3,
    minorgrid = true, minorgridlinewidth= 0.1, gridalpha = 0.18 , minorgridalpha = 0.21 , minorticks = true ,
    gridlinewidth = 0.6  ,  dpi = 300 , lw = 0.05)
# THESE PARAMS ARE TUNED FOR MINI_PLOTS !!!!!!



Gen_plot_B = histogram(centered_gen_hist[:,1] , label = "G band",  fill = true, fillalpha = 0.7, color = :blue, bins = 20, normalize = :pdf)
title!("Generated Lenses")
xaxis!((13 , 25))
Gen_plot_G = histogram(centered_gen_hist[:,2] , label = "R band",  fill = true, fillalpha = 0.7, color = :green, bins = 20, normalize = :pdf)
xaxis!((13 , 25))
Gen_plot_R = histogram(centered_gen_hist[:,3] , label = "I band",  fill = true, fillalpha = 0.7, color = :red, bins = 20, normalize = :pdf)
xaxis!("Magnitudes ",(13 , 25))

real_plot_B = histogram(centered_real_hist[:,1] , legend = false,  fill = true, fillalpha = 0.7, color = :blue, bins = 62, normalize = :pdf)
title!("Real candidates")
xaxis!((13 , 25))
real_plot_G = histogram(centered_real_hist[:,2] , legend = false,  fill = true, fillalpha = 0.7, color = :green, bins = 50, normalize = :pdf)
xaxis!((13 , 25))
real_plot_R = histogram(centered_real_hist[:,3] , legend = false,  fill = true, fillalpha = 0.7, color = :red, bins = 40, normalize = :pdf)
xaxis!("Magnitudes ",(13 , 25))

plot(real_plot_B , Gen_plot_B, real_plot_G , Gen_plot_G, real_plot_R   , Gen_plot_R, layout = (3,2))

savefig("Hist_comparaison.svg")



# Histograms all on one, not good fffor comparaison :
# histogram(centered_gen_hist[:,1] , label = "'Blue' channel",  fill = true, fillalpha = 1, color = :blue, bins = 30, normalize = :pdf)
# histogram!(centered_gen_hist[:,2] , label = "'Green' channel",  fill = true, fillalpha = 0.6, color = :green, bins = 30, normalize = :pdf)
# Gen_plot = histogram!(centered_gen_hist[:,3] , label = "'Red' channel",  fill = true, fillalpha = 0.3, color = :red, bins = 30, normalize = :pdf)
# title!("Generated Lenses")
# xaxis!("Magnitudes (no zero point)",(-17 , -9))
# yaxis!("PDF")
#
# histogram(centered_real_hist[:,1] , label = "'Blue' channel",  fill = true, fillalpha = 1, color = :blue, bins = 30, normalize = :pdf)
# histogram!(centered_real_hist[:,2] , label = "'Green' channel",  fill = true, fillalpha = 0.6, color = :green, bins = 30, normalize = :pdf)
# real_plot = histogram!(centered_real_hist[:,3] , label = "'Red' channel",  fill = true, fillalpha = 0.3, color = :red, bins = 30, normalize = :pdf)
# title!("Real candidates")
# xaxis!("Magnitudes (no zero point)",(-17 , -9))
# yaxis!("PDF")
#
# plot(real_plot,Gen_plot)
#
# Final_image_test = [colorview(RGB , permutedims( centered_gen_img[i,:,:,:] , [3,1,2])) for i = 1:length(centered_gen_img[:,1,1,1])]
# img = mosaicview(Final_image_test,fillvalue=.95, nrow=31, npad=1)
# Gen_plot_B = histogram(centered_gen_hist[:,1] , legend = false,  fill = true, fillalpha = 0.7, color = :blue, bins = 30, normalize = :pdf)
# title!("Generated Lenses")
# xaxis!((-17 , -9))
# Gen_plot_G = histogram(centered_gen_hist[:,2] , legend = false,  fill = true, fillalpha = 0.7, color = :green, bins = 30, normalize = :pdf)
# xaxis!((-17 , -9))
# Gen_plot_R = histogram(centered_gen_hist[:,3] , legend = false,  fill = true, fillalpha = 0.7, color = :red, bins = 30, normalize = :pdf)
# xaxis!("Magnitudes (no zero point)",(-17 , -9))
#
#
#
# real_plot_B = histogram(centered_real_hist[:,1] , legend = false,  fill = true, fillalpha = 0.7, color = :blue, bins = 30, normalize = :pdf)
# title!("Real candidates")
# xaxis!((-17 , -9))
# real_plot_G = histogram(centered_real_hist[:,2] , legend = false,  fill = true, fillalpha = 0.7, color = :green, bins = 30, normalize = :pdf)
# xaxis!((-17 , -9))
# real_plot_R = histogram(centered_real_hist[:,3] , legend = false,  fill = true, fillalpha = 0.7, color = :red, bins = 30, normalize = :pdf)
# xaxis!("Magnitudes (no zero point)",(-17 , -9))
#
#
# plot(real_plot_B , Gen_plot_B, real_plot_G , Gen_plot_G, real_plot_R   , Gen_plot_R, layout = (3,2))

# # Old version to show images.
#
# mosaic_length = 25
# offset = 0
# global plot_mosaic = []
#
# for i = 1:mosaic_length
#     push!(plot_mosaic, plot(Final_image[i+offset] ) )
#     plot!(labelfontsize =4, xtickfontsize=false, ytickfontsize=false)
# end
#
# plot(plot_mosaic[1], plot_mosaic[2], plot_mosaic[3], plot_mosaic[4],
#     plot_mosaic[5]  , plot_mosaic[6] , plot_mosaic[7], plot_mosaic[8] ,
#     plot_mosaic[9]  , plot_mosaic[10],plot_mosaic[11] ,plot_mosaic[12],
#     plot_mosaic[13], plot_mosaic[14], plot_mosaic[15] , plot_mosaic[16])
# savefig("Mosaic_testPixelCNN_CANDS_lots.png")
