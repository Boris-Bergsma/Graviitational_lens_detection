using LaTeXStrings
using PyCall
gc = pyimport("gc")

include("Training.jl")
include("Evaluation.jl")
include("Prediction.jl")
include("helpers.jl")


# Load images, Normalise_images, and create the train / test sets
load_length =  606
start_point = 0



lenses  = load_lenses_candidates(load_length, start_point)
Normalise_images(lenses)

Gan_lenses = load_lenses_Gen(load_length, start_point)
Normalise_images(Gan_lenses)

# Lenses_img  = load_lenses(length(background_img[:,1,1,1])-6200, start_point)
# Normalise_images(Lenses_img)
# Lenses_img = vcat(Gan_lenses,Lenses_img)

train_images, train_labels , test_images , test_labels = create_training_test(lenses,Gan_lenses, 0.7)

# CLear the variables to save on RAM
background_img = nothing
Lenses_img = nothing

gc.collect()

# logstrech(train_images)
# logstrech(test_images)

# Call the training,in python, wich returns the results of th training
acc, test = py"training_GEN"(train_images[:,:,:,:], train_labels , test_images[:,:,:,:] , test_labels  , 300 , false)
# acc, test = py"training"(train_images[:,:,:,:], train_labels , test_images[:,:,:,:] , test_labels  , 150 , false)

using Plots
theme(:wong)
pgfplotsx() # for detailed latex plots
default(titlefont = (20, "times"), legendfontsize = 12, guidefont = (20, :black),
    tickfont = (14, :black), framestyle = :box,legend = :topright, markerstrokewidth = 0.3,
    minorgrid = true, minorgridlinewidth= 0.2 ,gridalpha = 0.12 , minorgridalpha = 0.15, minorticks = true ,
    gridlinewidth = 1.5  ,  dpi = 100) # , gridalpha = 0.18 , minorgridalpha = 0.21 for svg export to pdf
# Plot the result in Julia
epochs = 1:length(acc)
plot(epochs,acc,lw =2, label = "Train accuracy", legend = :topright)
xlabel!("Epochs")
plot!(epochs,test,label= "Test accurcy",lw=2)
annotate!((length(acc)-50+4, 0.9, Plots.text("Callback",  16, color = :red)))
plot!((length(acc)-50).*ones(51), 0.5:0.01:1 , label = nothing,line = :dot, color = :red, lw = 1.5)
savefig("Plots/Learn_Gen_new.pdf") # Saves the plot from p as a .pdf vector graphic
# savefig("Plots/Learn_300_Gen.png")

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

predictions, fpr , tpr, auc_score = py"test"( Test_images , Test_labels , "Weights_B5_new_Gen.h5" )

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
savefig("Plots/AUC_curve_Gen.pdf")
savefig("Plots/AUC_curve_Gen.png")



# Convert the images to viewable objects.
Final_image = [colorview(RGB , permutedims( Lenses_img[i,:,:,:] , [3,1,2])) for i = 1:length(Lenses_img[:,1,1,1])]

using Images , ImageView

# Dsiplay mosaic of images.
mosaic_length = 16
offset = 6400
global plot_mosaic = Array{Any}(undef,16)
for i = 1:mosaic_length
    global plot_mosaic[i] = plot(Final_image[i+offset] )
    (plot!(labelfontsize =4, xtickfontsize=false, ytickfontsize=false))
end

plot(plot_mosaic[1], plot_mosaic[2], plot_mosaic[3], plot_mosaic[4],
    plot_mosaic[5]  , plot_mosaic[6] , plot_mosaic[7], plot_mosaic[8] ,
    plot_mosaic[9]  , plot_mosaic[10],plot_mosaic[11] ,plot_mosaic[12],
    plot_mosaic[13], plot_mosaic[14], plot_mosaic[15] , plot_mosaic[16])
