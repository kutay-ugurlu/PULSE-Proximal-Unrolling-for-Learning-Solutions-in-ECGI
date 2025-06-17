%% Load geometries
wipe

% Display all the geometries used in the Utah-490 ECGI studies
load('reordered_epigeom490sock_closed_aligned_shifted.mat') % heart
load('lungs_shifted.mat') % lungs_shifted is the new uncolliding one!
load('tank192_outw_struct.mat') % torso for the electrodes
load('tank771_closed2_outw_struct.mat') % fine torso - 771
load('selected_192_leads_frm_771.mat')
lungs_shifted.pts = lungs_shifted.vertices;
lungs_shifted.fac = lungs_shifted.faces;
lungs_shifted = rmfield(lungs_shifted,"vertices");
lungs_shifted = rmfield(lungs_shifted,"faces");

%% Save and reload 
surfaces = {};

%% Apply transform on heart

zview = [14 -76];
xview = [4.9 -71.5];
% dict = dictionary('x',[-37:0],'y',[-7:0],'z',[0:21]);
dict = dictionary(["x","y","z"],[-37,-7,21]);

for direction = ['xyz']
    angle = dict(string(direction));
    surfaces{1} = torso_tank;
    surfaces{2} = lungs_shifted;
    surfaces{end+1} = epigeom490sock_closed_aligned;
    surfaces{3}.pts = rotate_by(surfaces{3}.pts,direction,angle);
%         save(['surfaces_UTAH_HLT','_',direction,'_',num2str(angle),'.mat'],'surfaces')
%         
%         %% 
%         load(['surfaces_UTAH_HLT','_',direction,'_',num2str(angle),'.mat']);
    for i = 1:length(surfaces)
        surfaces{i}.pts = surfaces{i}.pts';
        surfaces{i}.fac = surfaces{i}.fac';
    end
    
    % conductivities (Ramanathan2001):
    sigma_out = 0;
    sigma_body = 0.0002; % S/mm = 0.2 S/m
    sigma_lung = 0.00005; % S/mm = 0.05 S/m
    sigma_muscle = 0.0013; % S/mm = 0.13 S/m
    
    % NOTE: You may need to convert these to S/m !! Check!
    % I am not certain about the ordering of the conductivities,
    % one of them should be outside of the surface, the other should be
    % inside of the surface conductivity!!
    surfaces{1}.sigma = [sigma_out sigma_body];
    surfaces{2}.sigma = [sigma_body sigma_lung];
    surfaces{3}.sigma = [sigma_body sigma_out];
    
    %%
    model_HLT = struct();
    model_HLT.surface = surfaces;
    surfaces(2) = []; 
    model_HT = struct();
    model_HT.surface = surfaces;
   
    %% Run the forward solver
    % % With the lungs:
    Trf_HLT_fine = -1*Forward(model_HLT,3);
    % % There may be a sign error in the forward solver.
    % % Check the body surface potentials to make sure they are correct.
    % % If necessary, multiply by -1
    
    % Homogeneous:
    Trf_HT_fine = -1*Forward(model_HT,2);
    
    % Generate Lux configurations from the fine meshed torsos
    selected_leads = load('selected_leads.mat').leads_from_771;
    Trf_HLT_coarse = Trf_HLT_fine(selected_leads,:); % 192'lik subset
    Trf_HT_coarse = Trf_HT_fine(selected_leads,:); % 192'lik subset
    save(['GeometricModels',filesep,'AlreadyReordered_HLT_',direction,'_',num2str(angle)],"Trf_HLT_coarse")
    save(['GeometricModels',filesep,'AlreadyReordered_HT_',direction,'_',num2str(angle)],"Trf_HT_coarse")

end

%% This part was for checking the 
% A = load('ForwMat_HLT.mat');
% A = A.Trf_HLT_leads;
% new_node_order = load('newnode_order_3.mat').node_order;
% A = A(:,new_node_order);
% norm(A-Trf_HLT_coarse)/norm(A)
% 
% A = load('ForwMat_HT.mat');
% A = A.Trf_HT_leads;
% new_node_order = load('newnode_order_3.mat').node_order;
% A = A(:,new_node_order);
% norm(A-Trf_HT_coarse)/norm(A)
%  