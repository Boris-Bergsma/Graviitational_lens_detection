using LaTeXStrings
using PyCall
gc = pyimport("gc")

include("Evaluation.jl")
include("Training.jl")
include("Prediction.jl")
include("helpers.jl")


# Load images, Normalise_images, and create the train / test sets
load_length =  7936
start_point = 0


Lenses_img = load_lenses(load_length, start_point)
Normalise_images(Lenses_img)
# logstrech(Lenses_img)

background_img  = load_Background(20000, 20000)
Normalise_images(background_img)
# logstrech(background_img)




# Lenses_img = load_lenses_mag_cut(load_length, start_point)
# Normalise_images(Lenses_img)


train_images, train_labels , test_images , test_labels = create_training_test(background_img,Lenses_img, 0.86)

# CLear the variables to save on RAM
background_img = nothing
Lenses_img = nothing

gc.collect()



# Call the training,in python, wich returns the results of th training
acc, test,  predictions, fpr , tpr, auc_score , tes_acc = py"adv_training"(train_images[:,:,:,:], train_labels , test_images[:,:,:,:] , test_labels  , 300 , false)
# acc, test,  predictions, fpr , tpr, auc_score , tes_acc = py"training_rgb "(train_images[:,:,:,:], train_labels , test_images[:,:,:,:] , test_labels  , 500 , false)

using Plots
theme(:wong2)
pgfplotsx() # for detailed latex plots
default(titlefont = (20, "times"), legendfontsize = 12, guidefont = (20, :black),
    tickfont = (14, :black), framestyle = :box,legend = :topright, markerstrokewidth = 0.3,
    minorgrid = true, minorgridlinewidth= 0.2 ,gridalpha = 0.12 , minorgridalpha = 0.15, minorticks = true ,
    gridlinewidth = 1.5  ,  dpi = 100) # , gridalpha = 0.18 , minorgridalpha = 0.21 for svg export to pdf
# Plot the result in Julia
epochs = 1:length(acc)
p = plot(epochs,acc,lw =2, label = "Train accuracy", legend = :bottomright)
xlabel!("Epochs")
plot!(epochs,test,label= "Test accurcy",lw=2)
Plots.annotate!((length(acc)-50+4, 0.7, Plots.text("Callback at $(round(tes_acc,digits=3))",  16, color = :red)))
plot!((length(acc)-50).*ones(51), 0.5:0.01:1 , label = nothing,line = :dot, color = :red, lw = 1.5)
savefig("Plots/Learn_adv.pdf") # Saves the plot from p as a .pdf vector graphic


#Evaluate model on a fresh test set, get the AUC score, and get predicitons
load_length = 1115
start_point = 6856

background_img  = load_Background(load_length, start_point)
Lenses_img  = load_lenses(load_length, start_point)

Normalise_images(background_img)
Normalise_images(Lenses_img)

Test_images = [background_img; Lenses_img ]
Test_labels = [zeros(load_length); ones(load_length)]

background_img = nothing
Lenses_img = nothing

predictions, fpr , tpr, auc_score = py"test"( Test_images , Test_labels , "Weights_B5_new.h5" )

using Plots
theme(:wong)
pgfplotsx() # for detailed latex plots
default(titlefont = (20, "times"), legendfontsize = 12, guidefont = (20, :black),
    tickfont = (14, :black), framestyle = :box,legend = :topright, markerstrokewidth = 0.3,
    minorgrid = true, minorgridlinewidth= 0.2 ,gridalpha = 0.12 , minorgridalpha = 0.15, minorticks = true ,
    gridlinewidth = 1.5  ,  dpi = 100) # , gridalpha = 0.18 , minorgridalpha = 0.21 for svg export to pdf

# Plot the roc curve in Julia
plot(fpr,tpr,lw =2, legendtitle = "Auc score : " * string(round(auc_score,digits = 4)),  fill = 0 ,  label = "ROC curve", fillalpha = 0.4,legend = :bottomright)
plot!(0:1,0:1, line = :dot  , lw = 3, label = "Random")
xlabel!("False positive rate")
ylabel!("True positive rate")
savefig("Plots/AUC_curve_adv.pdf")

# Convert the images to viewable objects.
Final_image = [colorview(RGB , permutedims( Test_images[i,:,:,:] , [3,1,2])) for i = 1:length(Test_images[:,1,1,1])]

using Images , ImageView

# Dsiplay mosaic of images.
mosaic_length = 16
offset = 0
global plot_mosaic = Array{Any}(undef,16)
for i = 1:mosaic_length
    if mod(i , 2) == 0
        global plot_mosaic[i] = plot(Final_image[i+offset] ,xlabel =  string(round(predictions[i+offset], digits = 3)*100) * "%" )
        (plot!(labelfontsize =4, xtickfontsize=false, ytickfontsize=false))
    else
        global plot_mosaic[i] = plot(Final_image[i+load_length+offset] , xlabel =  string(round(predictions[i+load_length+offset], digits = 3)*100) * "%"  )
        (plot!(labelfontsize =4, xtickfontsize=false, ytickfontsize=false))
    end
end

plot(plot_mosaic[1], plot_mosaic[2], plot_mosaic[3], plot_mosaic[4],
    plot_mosaic[5]  , plot_mosaic[6] , plot_mosaic[7], plot_mosaic[8] ,
    plot_mosaic[9]  , plot_mosaic[10],plot_mosaic[11] ,plot_mosaic[12],
    plot_mosaic[13], plot_mosaic[14], plot_mosaic[15] , plot_mosaic[16])
savefig("Plots/Mosaic_preds_final.pdf")


# Browse through DES to find lens cadidates :
load_length = 50000
start_point = 0

Test_images  = load_Background(load_length, start_point)
Normalise_images(Test_images)
logstrech(Test_images)

predictions = Array( py"predict"( Test_images ) )

range_test = 0:0.05:1
temp = zeros(length(range_test))
j = 0
for i = range_test
    global j += 1
    #Select images with a certain treshold
    treshold = i
    Candidates , percentage = Test_images[vec(predictions .> treshold), :,:,:], predictions[predictions .> treshold]
    temp[j] = length(percentage)
end

# Plot the roc curve in Julia
t = plot(range_test,temp .+ 0.01,lw =2, color = :blue, label = false, yscale = :log10)
xlabel!("Cuttof probability")
ylabel!("Number of lenses")
plot!(legend = :inside, xtickfontsize=10, ytickfontsize=10, xguidefontsize=16,yguidefontsize=16,legendfontsize=8,labelfontsize =8)
savefig("Plots/bestpercent.pdf")


treshold = 0.1
Candidates , percentage = Test_images[vec(predictions .> treshold), :,:,:], predictions[predictions .> treshold]
# Convert the images to viewable objects.
Final_image = [colorview(RGB , permutedims( Candidates[i,:,:,:] , [3,1,2])) for i = 1:length(Candidates[:,1,1,1])]



# Prediciton to compare adv learning.

background_img  = load_lenses_candidates(606, 0)
Normalise_images(background_img)

predict_no_adv = py"predict"( background_img ,  "Weights_B0_test.h5" )
predict_adv = py"predict"( background_img ,  "Weights_B0_adv.h5" )

threshold = 0.42
number_no_adv = sum(predict_no_adv.>threshold)
threshold = 0.92
number_adv = sum(predict_adv.>threshold)


using Plots
theme(:wong)
pgfplotsx() # for detailed latex plots
default(titlefont = (20, "times"), legendfontsize = 12, guidefont = (20, :black),
    tickfont = (14, :black), framestyle = :box,legend = :topright, markerstrokewidth = 0.3,
    minorgrid = true, minorgridlinewidth= 0.2 ,gridalpha = 0.12 , minorgridalpha = 0.15, minorticks = true ,
    gridlinewidth = 1.5  ,  dpi = 100) # , gridalpha = 0.18 , minorgridalpha = 0.21 for svg export to pdf
# Plot the result in Julia
histogram(predict_no_adv ,lw = 0.1, label = "Regular training , ($number_no_adv above 0.5)",  normalize = :pdf,  fill = true, fillalpha = 0.99,nbins = 100, legend = :topleft)
histogram!(predict_adv , lw = 0.1, label = "Adverserial training, ($number_adv above 0.5)",  normalize = :pdf , fill = true, fillalpha = 0.8,nbins = 100)
xaxis!("CNN scores")
yaxis!("PDF")
# lens!([0.25, 1], [0, 2], inset = (1, bbox(0.3, 0.2, 0.6, 0.6) ) )
lens!([0, 0.6], [0, 2], inset = (1, bbox(0.1, 0.2, 0.6, 0.6) ) )
savefig("Plots/hist_adv_cands.svg") # Saves the plot from p as a .pdf vector graphic


# Dsiplay mosaic of images.
mosaic_length = 9
offset = 40
global plot_mosaic = Array{Any}(undef,mosaic_length)

for i = 1:mosaic_length
    global plot_mosaic[i] = plot(Final_image[i+offset] ,xlabel =  string(round(percentage[i+offset], digits = 3)*100) * "%" )
    (plot!(labelfontsize = 4, xtickfontsize=false, ytickfontsize=false))
end

plot(plot_mosaic[1], plot_mosaic[2], plot_mosaic[3], plot_mosaic[4],
    plot_mosaic[5]  , plot_mosaic[6] , plot_mosaic[7], plot_mosaic[8] ,
    plot_mosaic[9]  )
savefig("Plots/Mosaic_Candidates_DES_cut1_log.pdf")

# plot the image samples (good way)
using Plots
using Images , ImageView
using MosaicViews, ImageShow
theme(:wong2)
pyplot()


Final_image = [colorview(RGB , permutedims( Lenses_img[i,:,:,:] , [3,1,2])) for i = 1:length(Lenses_img[:,1,1,1])]

# Dsiplay mosaic of images.
img = mosaicview(Final_image,fillvalue=.95, nrow=10, npad=1)
Images.save("Mosaic_logS.png",img)
