function Z_BH = Forward(model, noOfLayers)

switch noOfLayers
    
case 2
    
    P_HH = MakeP(model,2,2);
    G_HH = MakeG(model,2,2);
    P_HB = MakeP(model,2,1);
    
    P_BH = MakeP(model,1,2);
    G_BH = MakeG(model,1,2);
    P_BB = MakeP(model,1,1);
    
%    save P_HH;
    
    inv_G_HH = my_inv(G_HH);
    L1 = G_BH * inv_G_HH;
    
    Z_BH = my_inv(P_BB - L1 * P_HB) * (L1 * P_HH - P_BH);
    
case 3
    
    G_HH = MakeG(model,3,3);
    G_LH = MakeG(model,2,3);
    G_BH = MakeG(model,1,3);
    
    disp('invG_HH')
    inv_G_HH = my_inv(G_HH);
    clear G_HH
    disp('L1')
    L1 = G_BH * inv_G_HH;
    disp('L2')
    L2 = G_LH * inv_G_HH;
    
    clear G_BH G_LH inv_G_HH 
    
    P_LL = MakeP(model,2,2);
    P_HL = MakeP(model,3,2);
    
    disp('L3')
    L3 = my_inv(P_LL - L2 * P_HL);
    clear P_LL
    
    P_BL = MakeP(model,1,2);
    P_HB = MakeP(model,3,1);
    P_LB = MakeP(model,2,1);
    
    disp('L4')
    L4 = (P_BL - L1 * P_HL) ...
       * L3 ...
       * (L2 * P_HB - P_LB);
   
    clear P_LB
    
    P_HH = MakeP(model,3,3);
    P_LH = MakeP(model,2,3);
    
    disp('L5')
    L5 = (P_BL - L1 * P_HL) ...
       * L3 ...
       * (P_LH - L2 * P_HH);
    clear P_BL P_HL P_LH
      
    P_BH = MakeP(model,1,3);
    P_BB = MakeP(model,1,1);
           
    disp('Zbh')
    Z_BH = my_inv(L4 + (P_BB - L1 * P_HB)) * (L5 + (L1 * P_HH - P_BH));
    
    save separateMATR
case 4
    
    G_HH = MakeG(model,4,4);
    inv_G_HH = my_inv(G_HH);
    clear G_HH
    
    G_BH = MakeG(model,1,4);
    T1 = G_BH * inv_G_HH;
    clear G_BH
    
    G_LH = MakeG(model,3,4);
    G_SH = MakeG(model,2,4);
    
    G_TH = [G_LH ; G_SH];
    clear G_LH G_SH
    
    T2 = G_TH * inv_G_HH;
    clear G_TH inv_G_HH
    
    P_HL = MakeP(model,4,3);
    P_HS = MakeP(model,4,2);
    P_HT = [P_HL P_HS];
    clear P_HL P_HS
    
    P_LL = MakeP(model,3,3);
    P_LS = MakeP(model,3,2);  
    P_SL = MakeP(model,2,3);
    P_SS = MakeP(model,2,2);
    P_TT = [P_LL P_LS; P_SL P_SS];
    clear P_LL P_LS P_SL P_SS
    
    T3 = my_inv(P_TT - T2 * P_HT);
    clear P_TT
    
    P_BL = MakeP(model,1,3);
    P_BS = MakeP(model,1,2);
    P_BT = [P_BL P_BS];
    clear P_BL P_BS
    
    P_LB = MakeP(model,3,1);
    P_SB = MakeP(model,2,1);
    P_TB = [P_LB ; P_SB];
    clear P_LB P_SB
    
    P_HB = MakeP(model,4,1);
    T4 = (P_BT - T1 * P_HT) ...
       * T3 ...
       * (T2 * P_HB - P_TB);
    clear P_TB    
    
    P_LH = MakeP(model,3,4);
    P_SH = MakeP(model,2,4);
    P_TH = [P_LH ; P_SH];
    clear P_LH P_SH
    
    P_HH = MakeP(model,4,4);
    T5 = (P_BT - T1 * P_HT) ...
       * T3 ...
       * (P_TH - T2 * P_HH);
    
    clear T2 T3

    P_BH = MakeP(model,1,4);
    P_BB = MakeP(model,1,1);
           
    Z_BH = my_inv(T4 + (P_BB - T1 * P_HB)) * (T5 + (T1 * P_HH - P_BH));
    
otherwise
    error('Unrecognized number of layers');   
    
end

return  

function PP = MakeP(model,surf1,surf2)

    fprintf(1, 'Generating P matrix %d %d\n', surf1, surf2);

    Pts = model.surface{surf1}.pts;
    Pos = model.surface{surf2}.pts;
    Tri = model.surface{surf2}.fac;

    NumPts = size(Pts,2);
    NumPos = size(Pos,2);
    NumTri = size(Tri,2);

    % Define a unitary vector 
    In = ones(1,NumTri);

    GeoData = zeros(NumPts,NumTri,3);

    for p = 1:NumPts,
        
        % Define all triangles that are no autoangles

        if surf1 == surf2,
            Sel = find((Tri(1,:) ~= p)&(Tri(2,:)~=p)&(Tri(3,:)~=p));
        else
            Sel = 1:NumTri;
        end    
        
        % Define vectors for position p

        ym = Pts(:,p)*ones(1,NumTri);
        y1 = Pos(:,Tri(1,:))-ym;
        y2 = Pos(:,Tri(2,:))-ym;
        y3 = Pos(:,Tri(3,:))-ym;

        epsilon = 1e-12;

        gamma = zeros(3,NumTri);

        % Speeding up the computation by splitting the formulas
        y21 = y2 - y1;
        y32 = y3 - y2;
        y13 = y1 - y3;
        
        Ny1 = sqrt(sum(y1.^2));
        Ny2 = sqrt(sum(y2.^2));
        Ny3 = sqrt(sum(y3.^2));
        
        Ny21 = sqrt(sum((y21).^2));
        Ny32 = sqrt(sum((y32).^2));
        Ny13 = sqrt(sum((y13).^2));
        
        
        
        NomGamma = Ny1.*Ny21 + sum(y1.*y21);
        DenomGamma = Ny2.*Ny21 + sum(y2.*y21);

        W = find((abs(DenomGamma-NomGamma) > epsilon)&(DenomGamma ~= 0)&(NomGamma ~= 0));
        gamma(1,W) = -ones(1,size(W,2))./Ny21(W).*log(NomGamma(W)./DenomGamma(W));

        NomGamma = Ny2.*Ny32 + sum(y2.*y32);
        DenomGamma = Ny3.*Ny32 + sum(y3.*y32);

        W = find((abs(DenomGamma-NomGamma) > epsilon)&(DenomGamma ~= 0)&(NomGamma ~= 0));
        gamma(2,W) = -ones(1,size(W,2))./Ny32(W).*log(NomGamma(W)./DenomGamma(W));

        NomGamma = Ny3.*Ny13 + sum(y3.*y13);
        DenomGamma = Ny1.*Ny13 + sum(y1.*y13);

        W = find((abs(DenomGamma-NomGamma) > epsilon)&(DenomGamma ~= 0)&(NomGamma ~= 0));
        gamma(3,W) = -ones(1,size(W,2))./Ny13(W).*log(NomGamma(W)./DenomGamma(W));

        d = sum(y1.*cross(y2,y3));
        N = cross(y21,-y13);
        A2 = sum(N.*N);

        OmegaVec = [1 1 1]'*(gamma(3,:)-gamma(1,:)).*y1 + [1 1 1]'*(gamma(1,:)-gamma(2,:)).*y2 +[1 1 1]'*(gamma(2,:)-gamma(3,:)).*y3; %'

        Nn = (Ny1.*Ny2.*Ny3+Ny1.*sum(y2.*y3)+Ny3.*sum(y1.*y2)+Ny2.*sum(y3.*y1));

        Omega = zeros(1,NumTri);

        Vz = find(Nn(Sel) == 0);
        Vp = find(Nn(Sel) > 0); 
        Vn = find(Nn(Sel) < 0);
        if size(Vp,1) > 0, Omega(Sel(Vp)) = 2*atan(d(Sel(Vp))./Nn(Sel(Vp))); end;
        if size(Vn,1) > 0, Omega(Sel(Vn)) = 2*atan(d(Sel(Vn))./Nn(Sel(Vn)))+2*pi; end;
        if size(Vz,1) > 0, Omega(Sel(Vz)) = pi*sign(d(Sel(Vz))); end;

        zn1 = sum(cross(y2,y3).*N); 
        zn2 = sum(cross(y3,y1).*N);
        zn3 = sum(cross(y1,y2).*N);

        % Compute spherical angles
        GeoData(p,Sel,1) = In(Sel)./A2(Sel).*((zn1(Sel).*Omega(Sel)) + d(Sel).*sum((y32(:,Sel)).*OmegaVec(:,Sel))); % linear interp function corner 1
        GeoData(p,Sel,2) = In(Sel)./A2(Sel).*((zn2(Sel).*Omega(Sel)) + d(Sel).*sum((y13(:,Sel)).*OmegaVec(:,Sel))); % linear interp function corner 2
        GeoData(p,Sel,3) = In(Sel)./A2(Sel).*((zn3(Sel).*Omega(Sel)) + d(Sel).*sum((y21(:,Sel)).*OmegaVec(:,Sel))); % linear interp function corner 3

    end 
    
    PP = zeros(NumPts,NumPos);
    
    % Assume every line being multiplied by this amount. 
    
    C = (1/(4*pi))*(model.surface{surf2}.sigma(1) - model.surface{surf2}.sigma(2));

    for q=1:NumPos
    
        V = find(Tri(1,:)==q);
        for r = 1:size(V,2)
            PP(:,q) = PP(:,q) - C*GeoData(:,V(r),1);
        end;
      
        V = find(Tri(2,:)==q);
        for r = 1:size(V,2)
            PP(:,q) = PP(:,q) - C*GeoData(:,V(r),2);
        end;

        V = find(Tri(3,:)==q);
        for r = 1:size(V,2)
            PP(:,q) = PP(:,q) - C*GeoData(:,V(r),3); 
        end;
        
    end
    
    if surf1 == surf2
          
        for p = 1:NumPts
            PP(p,p) = -sum(PP(p,:))+model.surface{surf2}.sigma(1);
        end    
         
    end
return

function GG = MakeG(model,surf1,surf2)

    fprintf(1, 'Generating G matrix %d %d\n', surf1, surf2);
    
    Pts = model.surface{surf1}.pts;
    Pos = model.surface{surf2}.pts;
    Tri = model.surface{surf2}.fac;

    NumPts = size(Pts,2);
    NumPos = size(Pos,2);
    NumTri = size(Tri,2);

    GG = zeros(NumPts,NumPos);
    for k = 1:NumPts,
        W = bem_radon(Tri',Pos',Pts(:,k)');
        
        for m = 1:NumTri,
            GG(k,Tri(:,m)) = GG(k,Tri(:,m)) + (1/(4*pi))*W(m,:);
        end
    end
return
    
function W = bem_radon(TRI,POS,OBPOS);

% initial weights
s15 = sqrt(15);
w1 = 9/40;
w2 = (155+s15)/1200; w3 = w2; w4 = w2;
w5 = (155-s15)/1200; w6 = w5; w7 = w6;
s  = (1-s15)/7;
r  = (1+s15)/7;

% how many, allocate output
nrTRI = size(TRI,1);
nrPOS = size(POS,1);
W = zeros(nrTRI,3);

% move all positions to OBPOS as origin
POS = POS - ones(nrPOS,1)*OBPOS;
I = find(sum(POS'.^2)<eps); Ising = [];
if ~isempty(I), 
    Ising  = [];
    for p = 1:length(I);
        [tx,dummy] = find(TRI==I(p)); 
        Ising = [Ising tx'];
    end
end


% corners, center and area of each triangle
P1 = POS(TRI(:,1),:); 
P2 = POS(TRI(:,2),:); 
P3 = POS(TRI(:,3),:);
C = (P1 + P2 + P3) / 3;
N = cross(P2-P1,P3-P1);
A = 0.5 * sqrt(sum(N'.^2))';

% point of summation (positions)
q1 = C;
q2 = s*P1 + (1-s)*C;
q3 = s*P2 + (1-s)*C;
q4 = s*P3 + (1-s)*C;
q5 = r*P1 + (1-r)*C;
q6 = r*P2 + (1-r)*C;
q7 = r*P3 + (1-r)*C;

% norm of the positions
nq1 = sqrt(sum(q1'.^2))';
nq2 = sqrt(sum(q2'.^2))';
nq3 = sqrt(sum(q3'.^2))';
nq4 = sqrt(sum(q4'.^2))';
nq5 = sqrt(sum(q5'.^2))';
nq6 = sqrt(sum(q6'.^2))';
nq7 = sqrt(sum(q7'.^2))';

% weight factors for linear distribution of strengths
a1 = 2/3; b1 = 1/3;
a2 = 1-(2*s+1)/3; b2 = (1-s)/3;
a3 = (s+2)/3; b3 = (1-s)/3;
a4 = (s+2)/3; b4 = (2*s+1)/3;
a5 = 1-(2*r+1)/3; b5 = (1-r)/3;
a6 = (r+2)/3; b6 = (1-r)/3;
a7 = (r+2)/3; b7 = (2*r+1)/3;

% calculated different weights
W(:,1) = A.*((1-a1)*w1./nq1 + (1-a2)*w2./nq2 + (1-a3)*w3./nq3 + (1-a4)*w4./nq4 + (1-a5)*w5./nq5 + (1-a6)*w6./nq6 + (1-a7)*w7./nq7);
W(:,2) = A.*((a1-b1)*w1./nq1 + (a2-b2)*w2./nq2 + (a3-b3)*w3./nq3 + (a4-b4)*w4./nq4 + (a5-b5)*w5./nq5 + (a6-b6)*w6./nq6 + (a7-b7)*w7./nq7);
W(:,3) = A.*(b1*w1./nq1 + b2*w2./nq2 + b3*w3./nq3 + b4*w4./nq4 + b5*w5./nq5 + b6*w6./nq6 + b7*w7./nq7);

% do singular triangles!
for i=1:length(Ising),
	I = Ising(i);
	W(I,:) = bem_sing(POS(TRI(I,:),:));
end    
    
return    

function W = bem_sing(TRIPOS);
% W = bem_sing(TRIPOS);
%
% W(J) is the contribution at vertex 1 from unit strength
% at vertex J, J = 1,2,3

% find point of singularity and arrange tripos
ISIN = find(sum(TRIPOS'.^2)<eps);
if isempty(ISIN), error('Did not find singularity!'); return; end
temp = [1 2 3;2 3 1;3 1 2];
ARRANGE = temp(ISIN,:);

% Divide vertices in RA .. RC
% The singular node is called A, its cyclic neighbours B and C
RA = TRIPOS(ARRANGE(1),:);
RB = TRIPOS(ARRANGE(2),:);
RC = TRIPOS(ARRANGE(3),:); 

% Find projection of vertex A (observation point) on the line
% running from B to C
[RL,AP] = laline(RA,RB,RC);

% find length of vectors BC,BP,CP,AB,AC
BC = norm(RC-RB);
BP = abs(RL)*BC;
CP = abs(1-RL)*BC;
AB = norm(RB-RA);
AC = norm(RC-RA);

% set up basic weights of the rectangular triangle APB
% WAPB(J) is contribution at vertex A (== observation position!) 
% from unit strength in vertex J, J = A,B,C
if abs(RL) > eps,
	a = AP; 
	b = BP;
	c = AB;
	log_term = log( (b+c)/a );
	WAPB(1) = a/2 * log_term;
	w = 1-RL; 
	WAPB(2) = a* (( a-c)*(-1+w) + b*w*log_term )/(2*b);
	w = RL;
	WAPB(3) = a*w *( a-c  +  b*log_term )/(2*b);
else
	WAPB = [0 0 0];
end

% set up basic weights of the rectangular triangle APB
% WAPC(J) is contribution at vertex A (== observation position!) 
% from unit strength in vertex J, J = A,B,C
if abs(RL-1) > eps,
	a = AP;
	b = CP;
	c = AC;
	log_term = log( (b+c)/a );
	WAPC(1) = a/2 * log_term;
	w = 1-RL;
	WAPC(2) = a*w *( a-c  +  b*log_term )/(2*b);
	w = RL;
	WAPC(3) = a* (( a-c)*(-1+w) + b*w*log_term )/(2*b);
else
	WAPC = [ 0 0 0];
end

% Compute total weights taking into account the position P on BC
if RL<0, WAPB = -WAPB; end
if RL>1, WAPC = -WAPC; end
W = WAPB + WAPC;

% arrange back
W(ARRANGE) = W;

return
%%%%% end mono_ana %%%%%


%%%%% local function %%%%%%
function [rl,ap] = laline(ra,rb,rc);
% find projection P of vertex A (observation point) on the line
% running from B to C
% rl = factor of vector BC, position p = rl*(rc-rb)+rb
% ap = distance from A to P
%
% called by mono_ana (see above)

% difference vectors
rba = ra - rb;
rbc = rc - rb;

% normalize rbc
nrbc = rbc / norm(rbc) ;

% inproduct and correct for normalization
rl = rba * nrbc';
rl = rl / norm(rbc) ;

% point of projection
p = rl*(rbc)+rb;

% distance
ap = norm(ra-p);

%%%%% end laline %%%%%
return

function R = my_inv(A)

[L, U, P] = lu(A);
y = L\P;
R = U\y;    
%%%%% local function %%%%%%
return
