using Glob
using FITSIO
using Statistics


function load_images_filename(load_length = 10000, start_point = 0 )

    file_names = glob("DES*.fits","/home/boris/Documents/Master/Projet_CSE/Lens_classification/DES_cut")

    data = zeros(load_length,50,50,3)

    for i=1:load_length
        data_temp = FITS(file_names[i+start_point])
        data[i,:,:,1] = read(data_temp[1])
        data[i,:,:,2] = read(data_temp[2])
        data[i,:,:,3] = read(data_temp[3])
        close( data_temp )
    end

    return data , file_names[1:load_length]

 end

function load_Background(load_length = 10000, start_point = 0)

    file_names = glob("DES*.fits","/home/boris/Documents/Master/Projet_CSE/Lens_classification/DES_cut")

    data = zeros(load_length,50,50,3)

    for i=1:load_length
        data_temp = FITS(file_names[i+start_point])
        data[i,:,:,1] = read(data_temp[1])
        data[i,:,:,2] = read(data_temp[2])
        data[i,:,:,3] = read(data_temp[3])
        close( data_temp )
    end

    return data

 end

function load_lenses(load_length = 10000,  start_point = 0, foldername = "Lens_classification/Lenses_karina/")

     file_names = glob("*.fits","/home/boris/Documents/Master/Projet_CSE/"*foldername)
     print("/home/boris/Documents/Master/Projet_CSE/"*foldername)
     data = zeros(load_length,50,50,3)

     for i=1:load_length
         data_temp = FITS(file_names[i+start_point])
         data[i,:,:,1] = read(data_temp[1])
         data[i,:,:,2] = read(data_temp[2])
         data[i,:,:,3] = read(data_temp[3])
         close( data_temp )
     end

 return data

end

function load_lenses_CFIS(load_length = 10000,  start_point = 0, foldername = "cand_sim50-80/" )

     file_names = glob("*.fits","/home/boris/Documents/Master/Projet_CSE/GAN_lenses_generation/CFIS-candidates/"*foldername)

     data = zeros(load_length,44,44)
     couter = 0

     for i=1:load_length

         data_temp = FITS(file_names[i+start_point])

         if size(data_temp[1]) == (44,44)
             data[i,:,:] = read(data_temp[1])
             couter += 1
         end
         close( data_temp )
     end
 data = data[1:couter,:,:]
 return data

end


function logstrech(images, a = 1000)

    for i = 1:length(images[:,1,1,1])
        for j = 1:3
            images[i,:,:,j] = log.( a.*images[i,:,:,j] .+ 1 ) / log(a+1)
        end
    end

end

function augment_rot(images)

    img_90 = zeros(length(images[:,1,1,1]),50,50,3)
    img_180 = zeros(length(images[:,1,1,1]),50,50,3)
    img_l_90 = zeros(length(images[:,1,1,1]),50,50,3)


    for i = 1:length(images[:,1,1,1])
        for j = 1:3
            img_90[i,:,:,j] = rotr90(images[i,:,:,j])
            img_180[i,:,:,j] = rot180(images[i,:,:,j])
            img_l_90[i,:,:,j] = rotl90(images[i,:,:,j])
        end
    end
    return vcat(images , img_90 , img_180 , img_l_90)
end

function Normalise_images(images, std = false)

    if std
        for i = 1:length(images[:,1,1,1])
            for j = 1:3
                images[i,:,:,j] .-= mean( images[i,:,:,j] )
                images[i,:,:,j] ./= var( images[i,:,:,j] )
                temp = images[i,:,:,j]
                images[i,:,:,j] = ( temp .-  minimum(temp)) ./ (maximum(temp) -  minimum(temp) )
            end
        end
    else
        for i = 1:length(images[:,1,1,1])
            for j = 1:3
                temp = images[i,:,:,j]
                images[i,:,:,j] = ( temp .-  minimum(temp)) ./ (maximum(temp) -  minimum(temp) )
            end
        end
    end
end


function Normalise_images_CFIS(images)
    for i = 1:length(images[:,1,1])
        temp = images[i,:,:]
        images[i,:,:] = ( temp .-  minimum(temp)) ./ (maximum(temp) -  minimum(temp) )
    end
end


function create_training_test(background_img, Lenses_img, train_ratio = 0.9)

     final_length = length(background_img[:,1,1,1])+length(Lenses_img[:,1,1,1])
     data = zeros(final_length, 50 ,50 , 3 )
     labels = zeros(final_length)

         for i = 1:final_length
             if mod(i , 2) == 0
                 data[i,:,:,:] = background_img[Int32(i/2.0),:,:,:]
                 labels[i] = 0
             else
                 data[i,:,:,:] = Lenses_img[Int32((i+1)/2.0),:,:,:]
                 labels[i] = 1
             end
         end

     data_train , labels_train = data[1:Int64(floor(train_ratio*final_length)),:,:,:] , labels[1:Int64(floor(train_ratio*final_length))]
     data_test , labels_test = data[Int64(ceil(train_ratio*final_length)):end,:,:,:] , labels[Int64(ceil(train_ratio*final_length)):end]

     return data_train, labels_train, data_test , labels_test
end




function circular_cut(images , radius = 3)

    # We first create the filter, with ones at the location of the wanted data.
    filter = zeros(50,50)

    for i = 1:50
        for j = 1:50
            if sqrt( (i-25)^2 + (j-25)^2) < radius
                filter[i,j] = 1
            end
        end
    end

    return_images = zeros(length(images[:,1,1,1]), 50 , 50 , 3 )

    # Then the filter is appplied per ellement using Julia's pointwise multiplication "."
    for k = 1:length(images[:,1,1,1])
        for j = 1:3
            return_images[k,:,:,j] = images[k,:,:,j].*filter
        end
    end
    return return_images
end


function load_lenses_Gen(load_length = 10000,  start_point = 0)

     file_names = glob("generated*.fits","/home/boris/Documents/Master/Projet_CSE/GAN_lenses_generation/Output_lenses_safe/")

     data = zeros(load_length,50,50,3)

     for i=1:load_length
         data_temp = FITS(file_names[i+start_point])
         data[i,:,:,:] = read(data_temp[1])
         close( data_temp )
     end

 return data

end


function compute_mag_hist_RGB(images_to_hist)

    hist = zeros(length(images_to_hist[:,1,1,1]), 3 )

    for k = 1:length(images_to_hist[:,1,1,1])
        for j = 1:3
            hist[k,j] = -2.5*log(sum(images_to_hist[k,:,:,j])) + 30 # zero point offset
        end
    end
    return hist
end




function save_generated_images(images)

    max_saved = glob("generated*.fits","Output_lenses_safe")
    max_saved = length(max_saved)

    for i = 1:length(images[:,1,1,1])
        f = FITS("Output_lenses/generated_$(i+max_saved).fits", "w")
        write(f, Float32.(images[i,:,:,:]))
        close(f)
    end

end
