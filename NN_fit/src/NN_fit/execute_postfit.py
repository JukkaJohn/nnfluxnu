def postfit_execution():
    
    if postfit_criteria:
            train_indices = train_indices.reshape(1, -1)
            val_indices = val_indices.reshape(1, -1)

            level1 = level1[0]
            num_reps = np.shape(N_event_pred)[0]

            if validation_split != 0.0:
                train_indices = train_indices[0]
                val_indices = val_indices[0]
                train_indices = train_indices.astype(int)
                val_indices = val_indices.astype(int)

                N_event_pred_train = N_event_pred[:, train_indices]
                pred_train = pred[:, train_indices]

                N_event_pred_val = N_event_pred[:, val_indices]
                data_val = data[val_indices]
                pred_val = pred[:, val_indices]

                level1_val = level1[val_indices]

                val_indices = torch.tensor(val_indices)

                cov_matrix_val = cov_matrix[val_indices][:, val_indices]

            if num_output_layers == 1:

                def compute_postfit_criteria(neutrino_pdfs, N_event_pred, pred):
                    # if postfit_criteria:
                    closure_fit = Postfit()
                    neutrino_pdfs, N_event_pred, pred = closure_fit.apply_postfit_criteria(
                        chi_square_for_postfit, N_event_pred, neutrino_pdfs, pred
                    )
                    return (neutrino_pdfs, N_event_pred, pred)

                if postfit_criteria and validation_split != 0.0:
                    neutrino_pdfs, N_event_pred_train, pred_train = (
                        compute_postfit_criteria(
                            neutrino_pdfs, N_event_pred_train, pred_train
                        )
                    )
                if postfit_criteria and validation_split == 0:
                    neutrino_pdfs, N_event_pred, pred = compute_postfit_criteria(
                        neutrino_pdfs, N_event_pred, pred
                    )
            if num_output_layers == 2:

                def compute_postfit_criteria(
                    neutrino_pdfs_mu, neutrino_pdfs_mub, N_event_pred, pred
                ):
                    # if postfit_criteria:
                    closure_fit = Postfit()
                    neutrino_pdfs_mu, _, _ = closure_fit.apply_postfit_criteria(
                        chi_square_for_postfit, N_event_pred, neutrino_pdfs_mu, pred
                    )
                    neutrino_pdfs_mub, N_event_pred, pred = (
                        closure_fit.apply_postfit_criteria(
                            chi_square_for_postfit, N_event_pred, neutrino_pdfs_mub, pred
                        )
                    )

                if postfit_criteria and validation_split != 0.0:
                    compute_postfit_criteria(
                        neutrino_pdfs_mu, neutrino_pdfs_mub, N_event_pred_train, pred_train
                    )
                if postfit_criteria and validation_split == 0:
                    compute_postfit_criteria(
                        neutrino_pdfs_mu, neutrino_pdfs_mub, N_event_pred, pred
                    )

        if postfit_measures:
            with open(f"{dir_for_data}/{filename_postfit}", "w") as file:
                file.write(f"level 1 shift {i}:\n")
                file.write("postfit report faser sim fit:\n")
                file.write("100 replicas:\n")

            def compute_postfit_measures(cov_matrix, N_event_pred, data, level1, pred):
                compute_postfit = Measures(cov_matrix, pdf, N_event_pred)
                if fit_level != 0:
                    delta_chi = compute_postfit.compute_delta_chi(
                        data,
                        N_event_pred,
                        level1,
                        x_alphas.detach().numpy().squeeze(),
                    )
                    print(f"mean delta chi = {delta_chi}")
                    with open(f"{dir_for_data}/{filename_postfit}", "w") as file:
                        file.write(f"delta chi^2 = {delta_chi}:\n")

                    if num_output_layers == 1:
                        accuracy = compute_postfit.compute_accuracy(
                            x_alphas.detach().numpy().flatten(),
                            neutrino_pdfs,
                            pdf,
                            1,
                            pdf_set,
                            particle_id_nu,
                        )
                        print(f"accuracy = {accuracy}")
                        with open(f"{dir_for_data}/{filename_postfit}", "w") as file:
                            file.write(f"accuracy = {accuracy}:\n")
                    if num_output_layers == 2:
                        accuracy_nu = compute_postfit.compute_accuracy(
                            x_alphas.detach().numpy().flatten(),
                            neutrino_pdfs_mu,
                            pdf,
                            1,
                            pdf_set,
                            particle_id_nu,
                        )

                        accuracy_nub = compute_postfit.compute_accuracy(
                            x_alphas.detach().numpy().flatten(),
                            neutrino_pdfs_mub,
                            pdf,
                            1,
                            pdf_set,
                            particle_id_nub,
                        )

                        with open(f"{dir_for_data}/{filename_postfit}", "w") as file:
                            file.write(f"accuracy nu = {accuracy_nu}:\n")
                            file.write(f"accuracy nub = {accuracy_nub}:\n")

                # if fit_level != 3:
                phi = compute_postfit.compute_phi(data, chi_square_for_postfit)
                print(f"phi = {phi}")
                with open(f"{dir_for_data}/{filename_postfit}", "w") as file:
                    file.write(f"phi = {phi}:\n")

                if fit_level == 2:
                    bias_to_var = compute_postfit.compute_bias_to_variance(
                        data, pred, N_event_pred, len(N_event_pred)
                    )
                    print(f"bias to var = {bias_to_var}")
                    with open(f"{dir_for_data}/{filename_postfit}", "w") as file:
                        file.write(f"bias_to_var = {bias_to_var}:\n")

            if postfit_measures and validation_split != 0.0:
                compute_postfit_measures(
                    cov_matrix_val, N_event_pred_val, data_val, level1_val, pred_val
                )
            if postfit_measures and validation_split == 0.0:
                compute_postfit_measures(cov_matrix, N_event_pred, data, level1, pred)

            with open(f"{dir_for_data}/{filename_postfit}", "w") as file:
                file.write(f"mean chi^2 = {np.mean(chi_square_for_postfit)}:\n")
                # file.write(f"average training length = {np.mean(training_lengths)}:\n")
                file.write("settings used:\n")
                file.write(f"learning rate = {lr}:\n")
                file.write(f"weigth decay = {wd}:\n")
                file.write(f"max training lenght = {max_epochs}:\n")
                file.write(f"patience = {patience}:\n")

        with open(f"{dir_for_data}/chi_square.txt", "w") as f:
            np.savetxt(f, chi_squares, delimiter=",")

        with open(f"{dir_for_data}/chi_squares_for_postfit.txt", "w") as f:
            np.savetxt(f, chi_square_for_postfit, delimiter=",")

        with open(f"{dir_for_data}/events.txt", "w") as f:
            np.savetxt(f, N_event_pred, delimiter=",")

        # write to lhapdf grid
        template_path = "/opt/anaconda3/envs/test_lhapdf/share/LHAPDF/template_.info"
        path = f"/opt/anaconda3/envs/test_lhapdf/share/LHAPDF/{neutrino_pdf_fit_name_lhapdf}/{neutrino_pdf_fit_name_lhapdf}.info"
        set_index = int(np.random.rand() * 1e7)
        pdf_dict_central = {}
        pdf_dict_error = {}

        if num_output_layers == 1:
            customize_info_file(template_path, path, set_index, f"{particle_id_nu}", 2)
            mean_pdf = np.mean(neutrino_pdfs, axis=0)
            std_pdf = np.std(neutrino_pdfs, axis=0)
            path = f"/opt/anaconda3/envs/test_lhapdf/share/LHAPDF/{neutrino_pdf_fit_name_lhapdf}/{neutrino_pdf_fit_name_lhapdf}_0000.dat"
            pdf_dict_error[12] = mean_pdf
            pdf_dict_central[12] = std_pdf
            write_lhapdf_grid(x_vals, pdf_dict_central, path)
            path = f"/opt/anaconda3/envs/test_lhapdf/share/LHAPDF/{neutrino_pdf_fit_name_lhapdf}/{neutrino_pdf_fit_name_lhapdf}_0001.dat"
            write_lhapdf_grid(x_vals, pdf_dict_error, path)
        if num_output_layers == 2:
            customize_info_file(
                template_path, path, set_index, f"{particle_id_nu}, {particle_id_nub}", 2
            )
            mean_pdf_nu = np.mean(neutrino_pdfs_mu, axis=0)
            mean_pdf_nub = np.mean(neutrino_pdfs_mub, axis=0)
            std_pdf_nu = np.std(neutrino_pdfs_mu, axis=0)
            std_pdf_nub = np.std(neutrino_pdfs_mub, axis=0)
            path = f"/opt/anaconda3/envs/test_lhapdf/share/LHAPDF/{neutrino_pdf_fit_name_lhapdf}/{neutrino_pdf_fit_name_lhapdf}_0000.dat"

            pdf_dict_error[14] = mean_pdf_nu
            pdf_dict_error[-14] = mean_pdf_nub
            pdf_dict_central[14] = std_pdf_nu
            pdf_dict_central[-14] = std_pdf_nub
            write_lhapdf_grid(x_vals, pdf_dict_central, path)
            # write_lhapdf_grid(x_vals, mean_pdf_nub, path, particle_id_nub)
            path = f"/opt/anaconda3/envs/test_lhapdf/share/LHAPDF/{neutrino_pdf_fit_name_lhapdf}/{neutrino_pdf_fit_name_lhapdf}_0001.dat"
            write_lhapdf_grid(x_vals, pdf_dict_error, path)
            # write_lhapdf_grid(x_vals, std_pdf_nub, path)

        if chi_square_for_postfit.size != 0:
            with open(f"{dir_for_data}/pred.txt", "w") as f:
                np.savetxt(f, pred, delimiter=",")

            with open(f"{dir_for_data}/train_indices.txt", "w") as f:
                np.savetxt(f, train_indices, delimiter=",")
            with open(f"{dir_for_data}/val_indices.txt", "w") as f:
                np.savetxt(f, val_indices, delimiter=",")

            with open(f"{dir_for_data}/training_lengths.txt", "w") as f:
                np.savetxt(f, training_lengths, delimiter=",")


    if produce_plot:
        if num_output_layers == 1:
            from plot_comb_pdf_cl import plot

            sig_tot = np.sqrt(stat_error**2 + sys_error**2)
            plot(
                x_vals,
                neutrino_pdfs,
                data,
                N_event_pred,
                sig_tot,
                particle_id_nu,
                low_bin,
                high_bin,
                pdf,
                pdf_set,
                dir_for_data,
            )
        if num_output_layers == 2:
            from plot_nu_nub_cl import plot

            sig_tot = np.sqrt(stat_error**2 + sys_error**2)
            plot(
                x_vals,
                neutrino_pdfs_mu,
                neutrino_pdfs_mub,
                data,
                N_event_pred_nu,
                N_event_pred_nub,
                sig_tot,
                particle_id_nu,
                low_bin_mu,
                high_bin_mu,
                low_bin_mub,
                high_bin_mub,
                pdf,
                pdf_set,
                dir_for_data,
            )

