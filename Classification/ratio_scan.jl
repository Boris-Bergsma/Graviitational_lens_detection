using PyCall

include("Training.jl")
include("Evaluation.jl")
include("Prediction.jl")
include("helpers.jl")


# Load images, Normalise_images, and create the train / test sets
load_length =  1000
start_point = 0


# this will be the adaptive train set
background_img  = load_Background(load_length*20, start_point)
Normalise_images(background_img)

lenses = load_lenses(load_length, start_point)
Normalise_images(lenses)

# Then we have the set test set

background_img_test  = load_Background(5000, 35000)
Normalise_images(background_img_test)

lenses_test = load_lenses(500, 5000)
Normalise_images(lenses_test)

test_images = vcat(lenses_test,background_img_test)
test_labels = [ones(500) ; zeros(5000)]

# Then we have candidates
lenses_cand  = load_lenses_candidates(606, start_point)
Normalise_images(lenses_cand)


test_length = 10

auc_score   = zeros(test_length)
test_acc  = zeros(test_length)
pre  = zeros(test_length)
rec  = zeros(test_length)
number_cand  = zeros(test_length)


for i = 1:test_length

    #Adding 1000 images each time the counter increases
    train_images = vcat(lenses,background_img[1:2000*i,:,:,:])
    train_labels = [ones(1000) ; zeros(2000*i)]

    # Call the training,in python, wich returns the results of th training
    auc_score[i] , test_acc[i] , pre[i] , rec[i] ,number_cand[i] = py"training_scan"(train_images, train_labels , test_images[1:1000,:,:,:] , test_labels[1:1000], test_images, test_labels  ,lenses_cand , 50 , false)

end

# saving the results because my RAM explode :/
auc_score = [ 0.9271 0.9521 0.9528 0.9586 0.960 0.9553 0.96144 0.9611 0.9738 0.9716 ]
test_acc = [0.931 0.921 0.957 0.930 0.961 0.962 0.967 0.964 0.971 0.969]
pre = [ 0.782 0.758 0.871 0.882 0.892 0.898 0.929 0.9312 0.950 0.963]
rec = [ 0.866 0.889 0.867 0.869 0.856 0.838 0.845 0.851 0.853 0.844 ]
number_cand = [ 0.787 0.856 0.721  0.718 0.76 0.689 0.632 0.624 0.616 0.472]


using Plots
theme(:wong)
pgfplotsx() # for detailed latex plots
default(titlefont = (20, "times"), legendfontsize = 12, guidefont = (20, :black),
    tickfont = (14, :black), framestyle = :box,legend = :topright, markerstrokewidth = 0.3,
    minorgrid = true, minorgridlinewidth= 0.2 ,gridalpha = 0.12 , minorgridalpha = 0.15, minorticks = true ,
    gridlinewidth = 1.5  ,  dpi = 100) # , gridalpha = 0.18 , minorgridalpha = 0.21 for svg export to pdf
# Plot the result in Julia
ratio = [1/(1+2*k) for k = 1:10]
plot(ratio,vec(test_acc),lw =1.5, label = "Test accuracy", legend = :bottomright)
plot!(ratio,vec(auc_score),lw =1.5, label = "AUC score")
plot!(ratio,vec(pre),lw =1.5, label = "Precision")
plot!(ratio,vec(rec),lw =1.5, label = "Recall", color = :black)
plot!(ratio,vec(number_cand),lw =1.5, label = "Recovered candidates", color = :red)
Plots.annotate!(0.13, 0.55, Plots.text("Real lens ratio in test ",  16, color = :purple))
plot!(0.1.*ones(54), 0.47:0.01:1 , label = nothing,line = :dot, color = :purple, lw = 1.5)
xlabel!("Ratio of lenses in training")
savefig("Plots/scan_ratio_small.pdf") # Saves the plot from p as a .pdf vector graphic



# plot the image samples
using Plots
using Images , ImageView
using MosaicViews, ImageShow
theme(:wong)
pyplot()

Final_image = [colorview(RGB , permutedims( train_images[i,:,:,:] , [3,1,2])) for i = 1:1000]

# Dsiplay mosaic of images.
img = mosaicview(Final_image,fillvalue=.95, nrow=31, npad=1)
Images.save("Plots/lenses_candidates_100.png",img)


# This is a test to see if it is possible to check somthing

train_images, train_labels , test_images , test_labels = create_training_test(background_img .+0.0924563523971 ,Gan_lenses.+0.026454143016613224, 0.99)

test_images = vcat(lenses,background_img_test.+0.0924563523971)
test_labels = [ones(606) ; zeros(606)]

# Call the training,in python, wich returns the results of th training
pred, fpr, tpr , score[i], aucs_scores[i] = py"training_GEN"(train_images[:,:,:,:], train_labels , test_images[:,:,:,:] , test_labels  , 100 , false)
