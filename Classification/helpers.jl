using Glob
using FITSIO
using Statistics
using DelimitedFiles


function load_images_filename(load_length = 10000, start_point = 0 )

    file_names = glob("DES*.fits","DES_cut")

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

    file_names = glob("DES*.fits","DES_cut")

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

function load_lenses(load_length = 10000,  start_point = 0)

     file_names = glob("SIM*.fits","Lenses_karina")

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

function load_lenses_mag_cut(load_length = 10000,  start_point = 0)

     file_names = glob("SIM*.fits","Lenses_karina")
     mag_cuts = readdlm("sim_SL_uniV5_2.csv", ',')[2:end,1]

     mag_files = []
     for i = 1:load_length
         if lstrip("$(file_names[i][19:26])", '0') in string.(mag_cuts)
             push!(mag_files, file_names[i])
         end
     end

     load_length = length(mag_files)
     data = zeros(load_length,50,50,3)

     for i=1:load_length
         data_temp = FITS(mag_files[i+start_point])
         data[i,:,:,1] = read(data_temp[1])
         data[i,:,:,2] = read(data_temp[2])
         data[i,:,:,3] = read(data_temp[3])
         close( data_temp )
     end

 return data

end

function load_lenses_candidates(load_length = 10000,  start_point = 0)

      file_names = glob("*.fits","/home/boris/Documents/Master/Projet_CSE/GAN_lenses_generation/DES-candidates/")

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

function logstrech(images, a = 10)

    for i = 1:length(images[:,1,1,1])
        for j = 1:3
            images[i,:,:,j] = log.( a.*images[i,:,:,j] .+ 1 ) / log(a+1)
        end
    end

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

function create_luminance(images)

    lum_img = zeros(length(images[:,1,1,1]), 50,50)

        for i = 1:length(images[:,1,1,1])
            lum_img[i,:,:] = images[i,:,:,1] .+ images[i,:,:,2] .+ images[i,:,:,3]
        end

    return lum_img
end

function augment_rot(images)

    img_90 = zeros(length(images[:,1,1,1]),50,50,3)
    img_180 = zeros(length(images[:,1,1,1]),50,50,3)
    img_l_90 = zeros(length(images[:,1,1,1]),50,50,3)

    img_copy = images
    for i in length(images[:,1,1,1])
        for j = 1:3
            img_90[i,:,:,j] = rotr90(images[i,50,50,j])
            img_180[i,:,:,j] = rot180(images[i,50,50,j])
            img_l_90[i,:,:,j] = rotj90(images[i,50,50,j])
        end
    end
    return vcat(img_copy , img_90 , img_180 , img_l_90)
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
