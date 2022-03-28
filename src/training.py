import torch
import numpy as np
import os
import src.utils as utils

import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from src.utils import save_dict_to_json, check_exist_and_create, check_and_make_dir
from torch.utils.tensorboard import SummaryWriter
# from src.dataset.gan_helper import gan_helper
from src.dataset.gan_helper_ngsim import gan_helper

import time

from tqdm import tqdm
from scipy.interpolate import griddata
from tqdm import tqdm

import sys

def FD_plot(FD_learner,FD_result_path,epoch):
    FD_learner.eval()
    rho_input=torch.arange(start=0.0,end=1.2,step=0.01).reshape((-1,1))
    FD_output=FD_learner(rho_input)
    plt.plot(rho_input.cpu().detach().numpy(),FD_output.cpu().detach().numpy())
    plt.xlabel('rho')
    plt.title('FD learner after {} epochs'.format(epoch))
    plt.savefig(FD_result_path+'FD_after_{}_epoch'.format(epoch),
                dpi=300,
                bbox_inches="tight")
    plt.close()
    FD_learner.train()

def training(model, optimizer, discriminator, train_feature, train_target, train_feature_phy, device,
             loops = None,
             noise_scale = None,
             physics=None,
             physics_optimizer=None,
             FD_plot_freq=None,
             restore_from=None,
             epochs=1000,
             batch_size=None,
             experiment_dir=None,
             save_frequency=10,
             verbose_frequency=1,
             verbose_computation_time=0,
             save_each_epoch="False",
             training_gan = True,
             training_gan_data = False,
             slice_at = None
             ):
    # Initialize tf.Saver instances to save weights during metrics_factory
    X_train = train_feature
    y_train = train_target
    X_train_phy = train_feature_phy
    begin_at_epoch = 0
    writer = SummaryWriter(os.path.join(experiment_dir, "summary"))
    weights_path = os.path.join(experiment_dir, "weights")
    check_exist_and_create(weights_path)
    if FD_plot_freq is not None:
        FD_result_path=os.path.join(experiment_dir, "FD_result/")
        check_exist_and_create(FD_result_path)

    if restore_from is not None:
        assert os.path.isfile(restore_from), "restore_from is not a file"
        # restore model
        begin_at_epoch = utils.load_checkpoint(restore_from, model, optimizer, begin_at_epoch) #torch.device('cpu')
        if physics is not None :
            str_idx = restore_from.index('.tar')
            restore_from_physics=restore_from[:str_idx] + '.physics' + restore_from[str_idx:]  
            #assert os.path.isfile(restore_from_physics), "restore_from_physics is not a file"
            #utils.load_checkpoint(restore_from_physics, physics,physics_optimizer, begin_at_epoch)
        logging.info(f"Restoring parameters from {restore_from}, restored epoch is {begin_at_epoch:d}")


    begin_at_epoch = 0

    best_loss = 10000
    best_last_train_loss = {"best":
                                {"loss": 100,
                                 "epoch": 0},
                            "last":
                                {"loss": 100,
                                 "epoch": 0},
                            }
    Data_loss = []
    np.random.seed(1)
    n_critic=10
    # shuffle the data
    idx = np.random.choice(X_train.shape[0], X_train.shape[0], replace=False)
    idx_phy = np.random.choice(X_train_phy.shape[0], X_train_phy.shape[0], replace=False)
    X_train = X_train[idx, :]
    y_train = y_train[idx, :]
    X_train_phy = X_train_phy[idx_phy, :]

    phys_loss_scale_np = 100
    for epoch in tqdm(range(begin_at_epoch, epochs)):
        



        #### hard code
        # y_train[:,0] = y_train[:,1]
        model_params_print = [p for p in model.parameters()]
        # physics_params_print = [p for p in physics.FD_learner.parameters()]
        # print("\n")
        # print("one flow parameter:    ", model_params_print[2].detach().cpu().numpy()[-1])
        # print("one physics parameter:    ", physics_params_print[2].detach().cpu().numpy()[-1])

        ####

        # train step
        # num_steps = X_train.shape[0] // batch_size

        num_steps = 100  #! HARDCODE HERE!#
        thresh = 0.0

        # batch_size_phy = X_train_phy.shape[0] // num_steps
        batch_size_phy = batch_size
        for step in range(num_steps):
            start_time = time.time()
            # x_batch = X_train[step * batch_size:(step + 1) * batch_size, :]
            # y_batch = y_train[step * batch_size:(step + 1) * batch_size, :]
            random_idx = np.random.choice(X_train.shape[0], batch_size, replace = False)
            x_batch = X_train[random_idx, :]
            y_batch = y_train[random_idx, :]
            # random sample X_train_phy
            random_idx = np.random.choice(X_train_phy.shape[0], batch_size_phy, replace=False)
            x_batch_phy = X_train_phy[random_idx, :]
            loss, activation = model.log_prob(y_batch, x_batch)
            loss = -loss.mean()
            data_loss = loss
            # print("loading_batch_time:", time.time() - start_time); start_time = time.time()
            # data_loss = torch.tensor(0)  #for pure gan debug
            if training_gan_data:
                fake_y = model.test(torch.from_numpy(x_batch).to(device))
                # T_fake = model.model_D(torch.cat((fake_y, torch.from_numpy(x_batch).to(device)), 1))
                T_fake = model.model_D(fake_y)
                if epoch >= model.switch_epoch:
                    if epoch%n_critic ==0:
                        loss_data_G = model.training_gan(y_batch, x_batch, writer, epoch, train=True)  # True: Train D
                    else:
                        loss_data_G = model.training_gan(y_batch, x_batch, writer, epoch, train=False)
                else:
                    if T_fake.mean()> -0.01 and model.n_G< 5:
                    # if model.n_G < 5:
                        loss_data_G = model.training_gan(y_batch, x_batch, writer, epoch, train = False) # False: Do not train D
                        model.n_G += 1
                    else:
                        loss_data_G = model.training_gan(y_batch, x_batch, writer, epoch, train= True) # True: Train D
                        model.n_G = 0
                # loss += 10*loss_data_G
                loss = loss_data_G
                loss_data_G_np = loss_data_G.cpu().detach().numpy()
            data_loss_np = 0
            loss_data_time = time.time()-start_time


            if (physics is not None) & (epoch//100%1 == 0) & (phys_loss_scale_np>thresh):
                physics_optimizer.zero_grad()

            # get physics_loss


            if (physics is not None) & (epoch//100%1 == 0) & (phys_loss_scale_np>thresh):
                start_time = time.time()
                phy_loss, physics_params, grad_hist = physics.get_residuals(model, x_batch_phy)
                phy_loss = phy_loss.mean()
                loss_phy_time = time.time() - start_time
                # print(physics_params["tau"])

                # try the normalizing loss
                # loss = loss * physics.hypers["alpha"]
                # loss += (1 - physics.hypers["alpha"]) * phy_loss
                data_loss_scale = torch.abs(loss.detach()) + torch.tensor(1e-6, dtype=torch.float32)
                phys_loss_scale = phy_loss.detach() + torch.tensor(1e-6, dtype=torch.float32)
                loss = loss * physics.hypers["alpha"] * phys_loss_scale/data_loss_scale
                loss += (1 - physics.hypers["alpha"]) * phy_loss

                phys_loss_scale_np = phys_loss_scale.cpu().numpy()
                # try the normalizing loss


                phy_loss_np = phy_loss.cpu().detach().numpy()

            if training_gan is True:
                # train the discriminator

                for _ in range(1):
                    if epoch%1 == 0:
                        discriminator.optimizer.zero_grad()
                        Gan_helper = gan_helper(noise_scale, loops, slice_at)
                        Ground_truth_figure,Ground_truth_figure_origin = Gan_helper.load_ground_truth()
                        rho_u_test = model.test(torch.from_numpy(Gan_helper.X_T_low_d).to(device))
                        generator_figure,_ = Gan_helper.reshape_to_figure(rho_u_test.detach()[:,0],
                                                                        rho_u_test.detach()[:,1])
                        T_real = discriminator.forward(torch.from_numpy(Ground_truth_figure).to(device))
                        T_fake = discriminator.forward(generator_figure)
                        loss_d = - (torch.log(1-torch.sigmoid(T_real)+1e-8) + torch.log(torch.sigmoid(T_fake)+1e-8))
                        loss_d.backward(retain_graph=True)
                        loss_d_np = loss_d.cpu().detach().numpy()
                        discriminator.optimizer.step()
                        del([loss_d, T_real, T_fake, rho_u_test, generator_figure])

                # Loss of the generator
                for _ in range(3):
                    optimizer.zero_grad()
                    rho_u_test = model.test(torch.from_numpy(Gan_helper.X_T_low_d).to(device))
                    generator_figure, generator_figure_origin = Gan_helper.reshape_to_figure(rho_u_test.detach()[:, 0],
                                                                    rho_u_test.detach()[:, 1])
                    generator_figure_np = generator_figure.cpu().detach().numpy()
                    T_fake = discriminator.forward(generator_figure)
                    T_real = discriminator.forward(torch.from_numpy(Ground_truth_figure).to(device))
                    loss_g = T_fake

                    loss_g_mse = torch.square(torch.from_numpy(Ground_truth_figure_origin[:,:,:,[0,8,12,20]]).to(device)-
                                              generator_figure_origin[:,:,:,[0,8,12,20]]).mean()
                    loss_g_mse_viz = torch.square(torch.from_numpy(Ground_truth_figure_origin).to(device)-
                                              generator_figure_origin).mean() # for visualization
                    T_real_np = T_real.cpu().detach().numpy()
                    loss_g_np = loss_g.cpu().detach().numpy()
                    # loss += loss_g_mse  + loss_g.squeeze().squeeze()
                    loss += loss_g.squeeze().squeeze()
                    writer.add_scalar(f"physics_grad/loss_g_mse", loss_g_mse, epoch + 1)
                    writer.add_scalar(f"physics_grad/loss_g", loss_g_np, epoch + 1)
                    writer.add_scalar(f"physics_grad/loss_g_mse_viz", loss_g_mse_viz, epoch + 1)



                    # loss += 10*loss_g.squeeze().squeeze()


                    # loss.backward(retain_graph=True)
                    # loss.backward()
                    # optimizer.step()

            start_time = time.time()
            #loss.backward(retain_graph=True)
            loss.backward(retain_graph=True)
            backward_all_time = time.time() - start_time
            # print("backward_time:", time.time() - start_time)

            start_time = time.time()
            if model.train is True:
                optimizer.step()
                optimizer.zero_grad()
                pass

            step_data_time = time.time() - start_time
            # print("step_time:", step_data_time)
            start_time = time.time()

            start_time = time.time()
            if (physics is not None) & (epoch//100%1 == 0)&(phys_loss_scale_np>thresh):
                if physics.train is True:
                    physics_optimizer.step()
                    #pass
            step_phy_time = time.time() - start_time

            if verbose_computation_time == 1:
                print(f"step = {epoch*num_steps + step:d}")
                print(f"loss_data_time: {loss_data_time:.5f}")
                print(f"loss_phys_time: {loss_phy_time:.5f}")
                print(f"backward_all_time: {backward_all_time:.5f}")
                print(f"step_data_time: {step_data_time:.5f}")
                print(f"step_phy_time: {step_phy_time:.5f}")
            # evaluation
            start_time = time.time()

            # save the data loss
            Data_loss.append(data_loss.cpu().detach().numpy())
            # print("evaluation_time:", time.time() - start_time)

            # delete the output tensor
            del([data_loss, loss])
            if (physics is not None) & (epoch//100%1 == 0)&(phys_loss_scale_np>thresh):
                del(phy_loss)

            # if training_gan_data:
            #     del(loss_data_G)

            if training_gan is True:
                del(loss_g)


            # below is for debug
            # a = activation_eval["x1_eval"].detach().cpu().numpy()
            # idx = np.argsort(a)[-1]
            # 4947
            # a = 1

            # below is to force the iteration # for each epoch to be 1.
            break


        # logging
        # if verbose_frequency > 0:
        #     if epoch % verbose_frequency == 0:
        #         logging.info(f"Epoch {epoch + 1}/{epochs}    loss={Data_loss[-1]:.3f}")

        # saving at every "save_frequency" or at the last epoch
        # Data_loss = [-10] #train gan data and commit out

        if (epoch % save_frequency == 0) | (epoch == begin_at_epoch + epochs - 1):
            activation_eval = model.eval(x_batch)
            start_time = time.time()
            is_best = Data_loss[-1] < best_loss
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'state_dict': model.state_dict(),
                                   'optim_dict': optimizer.state_dict()},
                                  is_best=is_best,
                                  checkpoint=weights_path,
                                  save_each_epoch=save_each_epoch)
            if (physics is not None) & (epoch//100%1 == 0):
                utils.save_checkpoint_physics({'epoch': epoch + 1,
                                   'state_dict': physics.state_dict(),
                                   'optim_dict': physics_optimizer.state_dict()},
                                  is_best=is_best,
                                  checkpoint=weights_path,
                                  save_each_epoch=save_each_epoch)

            # if best loss, update the "best_last_train_loss"
            if is_best:
                best_loss = Data_loss[-1]
                best_last_train_loss["best"]["loss"] = best_loss
                best_last_train_loss["best"]["epoch"] = epoch+1

            # update and save the latest "best_last_train_loss"
            best_last_train_loss["last"]["loss"] = Data_loss[-1]
            best_last_train_loss["last"]["epoch"] = epoch+1

            save_path = os.path.join(experiment_dir, "best_last_train_loss.json")
            save_dict_to_json(best_last_train_loss, save_path)

            # save loss to tensorboard
            writer.add_scalar("loss/train", Data_loss[-1], epoch+1)
            # writer.add_scalar("loss/train_data_loss", data_loss_np, epoch+1)
            if training_gan_data:
                writer.add_scalar("loss/train_Gan_loss(data)", loss_data_G_np, epoch+1)
            if physics is not None:
                writer.add_scalar("loss/train_phy_loss", phy_loss_np, epoch+1)
            if training_gan:
                writer.add_scalar("loss/train_Gan_Generator", loss_g_np, epoch + 1)
                writer.add_scalar("loss/train_Gan_Discriminator", loss_d_np, epoch + 1)



            # save activation to tensorboard
            for k, v in activation.items():
                writer.add_histogram(f"activation_train/{k:s}", v, epoch+1)
            for k, v in activation_eval.items():
                writer.add_histogram(f"activation_eval/{k:s}", v, epoch+1)
            if (physics is not None) & (epoch//100%1 == 0):
                # write the physics_params
                for k, v in physics_params.items():
                    if k=='tau':
                        v = v/50.0
                    if k=='nu':
                        v = v ##ground truth ==1
                    writer.add_scalar(f"physics_params/{k:s}", v.mean(), epoch+1)


                # write the hist of the gradient w.r.t x and t
                for k, v in grad_hist.items():
                    writer.add_histogram(f"grad/{k:s}", v, epoch+1)

                # for k, v in physics.torch_meta_params.items():
                #     if physics.meta_params_trainable[k] == "True":
                #         writer.add_scalar(f"physics_grad/dLoss_d{k:s}", v.grad, epoch + 1)

            # print("recording_time:", time.time()-start_time)
        if FD_plot_freq is not None:
            if (epoch % FD_plot_freq == 0) | (epoch == begin_at_epoch + epochs - 1) | (epoch == 0):
                FD_plot(physics.FD_learner,FD_result_path,epoch)



    # plot the abnormal training data loss
    if np.sum( np.where(np.array(Data_loss) > 0) ) > 10*num_steps:
        plt.plot(np.arange(len(Data_loss)), Data_loss)
        plt.xlabel("epoch")
        plt.ylabel("train loss")
        plt.savefig(os.path.join(experiment_dir,"abnormal_train_data_loss.png"))



def test(model, test_feature, test_target,
                restore_from=None,
                metric_functions = None,
                n_samples = None,
                noise = 0.2,
                args=None):
    # experiment_dir: where the model json file locates
    # Initialize tf.Saver instances to save weights during metrics_factory


    if restore_from is not None:
        assert os.path.isfile(restore_from), "restore_from is not a file"
        # restore model
        begin_at_epoch=0
        begin_at_epoch = utils.load_checkpoint(restore_from, model,optimizer=None,epoch= begin_at_epoch,
                                               device=model.device)
        logging.info(f"Restoring parameters from {restore_from}, restored epoch is {begin_at_epoch:d}")

    # make prediction
    #pre_train = not train_PUNN
    #test_feature, test_target = data_test #need to load test data...
    ###
    #test_prediction = model.predict(test_feature, n_repeat=n_samples, pre_train=pre_train)

    samples_mean_rho = np.zeros(( test_feature.shape[0], n_samples )) #115200,n_samples
    samples_mean_u = np.zeros(( test_feature.shape[0], n_samples ))
    torch.manual_seed(1)
    np.random.seed(1)
    for i in tqdm(range(0, n_samples )):
      result = model.test(torch.from_numpy(test_feature))
      rho_tensor = result[:,:1]
      u_tensor = result[:,1:]
      samples_mean_rho[:,i:i+1],samples_mean_u[:,i:i+1] = rho_tensor.detach().numpy(), u_tensor.detach().numpy()
    
    rho_star=test_target[:,0][:,None]
    u_star=test_target[:,1][:,None]
    exact_sample_rho = rho_star
    exact_sample_u = u_star
    b = exact_sample_rho
    c = exact_sample_u
    
    for i in range(n_samples -1):
        bb = b + noise * np.random.randn(rho_star.shape[0],1)
        exact_sample_rho = np.hstack((exact_sample_rho, bb))
        cc = c + noise * np.random.randn(u_star.shape[0],1)
        exact_sample_u = np.hstack((exact_sample_u, cc))
    # Compare mean and variance of the predicted samples as prediction and uncertainty
    RHO_pred = np.mean(samples_mean_rho, axis = 1) ##115200,1
    U_pred = np.mean(samples_mean_u, axis = 1) ##115200,1
    test_prediction=np.concatenate([RHO_pred[:,None],U_pred[:,None]],1) ## 115200,2
    #test_target #115200,2
    """
    RHO_pred = griddata( test_feature, RHO_pred.flatten(), (X, T), method='cubic')
    U_pred = griddata(test_feature, U_pred.flatten(), (X, T), method='cubic')
    """


    ###
    # exact_sample_u #115200,n_samples
    # exact_sample_rho #115200,n_samples
    # samples_mean_u #115200,n_samples
    # samples_mean_rho #115200,n_samples
    ###


    # convert to numpy
    #test_target = test_target.numpy() #115200,2
    #test_prediction = test_prediction.numpy() #115200,2

    metrics_dict = dict()
    kl = None
    kl_u = None
    kl_rho = None
    for k, func in metric_functions.items():
        if k == 'nlpd':
            use_mean = True if args.nlpd_use_mean == "True" else False
            metrics_dict[k] = [func(test_target, test_prediction,
                                   use_mean = use_mean,
                                   n_bands = args.nlpd_n_bands)]            

        elif (k == "kl") or (k == "fake_kl"):
            
            ###BCE with logit:
            #kl = func(torch.from_numpy(test_target), torch.from_numpy(test_prediction)).item()
            #metrics_dict[k] =[kl]

            ###get_KL:
            kl_rho=func(exact_sample_rho, samples_mean_rho)
            key_rho=k+'_rho'
            metrics_dict[key_rho] = [np.mean(kl_rho)]
            kl_u = func(exact_sample_u, samples_mean_u)
            key_u=k+'_u'
            metrics_dict[key_u] = [np.mean(kl_u)]

        elif k == "nll":
            log_prob, _ = model.log_prob(test_target, test_feature)
            nll = -log_prob.mean()
            metrics_dict[k] = [nll]
        else:
           metrics_dict[k+'_rho'] = [func(torch.from_numpy(test_target[:,0]), torch.from_numpy(test_prediction[:,0])).item()]
           metrics_dict[k+'_u'] = [func(torch.from_numpy(test_target[:,1]), torch.from_numpy(test_prediction[:,1])).item()]
        print('{}: done'.format(k))

    return metrics_dict, test_prediction, exact_sample_u,exact_sample_rho, samples_mean_u,samples_mean_rho,kl_u,kl_rho
    



#test(model,test_feature,test_label,   restore_from=restore_from,metric_functions=metric_fns,n_samples=args.test_sample,noise=args.noise,params=params)

def test_multiple_rounds(model, test_feature,test_label, test_rounds=1,
                         save_dir = None,
                         model_alias = None,                        
                **kwargs):

    metrics_dict, test_prediction, exact_sample_u,exact_sample_rho, samples_mean_u,samples_mean_rho, kl_u,kl_rho = test(model, test_feature,test_label,

                                         **kwargs)
    logging.info("Restoring parameters from {}".format(kwargs["restore_from"]))
    if test_rounds > 1:
        for i in range(test_rounds-1):
            metrics_dict_new= test(model, test_feature,test_label, 
                                         **kwargs) [0]
            for k in metrics_dict.keys():
                metrics_dict[k] += metrics_dict_new[k]
    
    check_and_make_dir(os.path.join(save_dir, model_alias))
    save_path_metric = os.path.join(save_dir, model_alias,
                                    f"metrics_test.json")
    #save_path_prediction = os.path.join(save_dir, model_alias,f"predictions_test.csv")
    #save_path_feature = os.path.join(save_dir, model_alias,f"features_test.csv")
    #save_path_target = os.path.join(save_dir, model_alias,f"targets_test.csv")
    #save_path_kl = os.path.join(save_dir, model_alias,f"kl_test.csv")

    save_path_prediction_rho = os.path.join(save_dir, model_alias,
                                        f"predictions_test_rho.csv")
    save_path_prediction_u = os.path.join(save_dir, model_alias,
                                        f"predictions_test_u.csv")
    save_path_feature = os.path.join(save_dir, model_alias,
                                        f"features_test.csv")
    save_path_target_rho = os.path.join(save_dir, model_alias,
                                        f"targets_test_rho.csv")
    save_path_target_u = os.path.join(save_dir, model_alias,
                                        f"targets_test_u.csv")
    save_path_kl_rho = os.path.join(save_dir, model_alias,f"kl_rho_test.csv")
    save_path_kl_u = os.path.join(save_dir, model_alias,f"kl_u_test.csv")
    
    save_dict_to_json(metrics_dict, save_path_metric)
    #np.savetxt(save_path_prediction, test_prediction, delimiter=",")
    #np.savetxt(save_path_feature, test_feature, delimiter=",")
    #np.savetxt(save_path_target, test_label, delimiter=",")
    #np.savetxt(save_path_kl, kl, delimiter=",")
    np.savetxt(save_path_prediction_rho, samples_mean_rho , delimiter=",")
    np.savetxt(save_path_prediction_u, samples_mean_u, delimiter=",")
    np.savetxt(save_path_feature, test_feature, delimiter=",")
    np.savetxt(save_path_target_rho, exact_sample_rho, delimiter=",")
    np.savetxt(save_path_target_u, exact_sample_u, delimiter=",")
    np.savetxt(save_path_kl_rho, kl_rho, delimiter=",")
    np.savetxt(save_path_kl_u, kl_u, delimiter=",")



