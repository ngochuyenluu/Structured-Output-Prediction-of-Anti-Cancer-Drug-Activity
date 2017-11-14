


% function for running MMCRF algorithm

function rtn = run_MMCRF(C, CGD, LBP, MAXITER, kernelfile, targetfile, edgefile, suffix, resultfile)
	C = str2num(C);
	CGD = str2num(CGD);
	LBP = str2num(LBP);
	MAXITER = str2num(MAXITER);

	global Kx_tr;
	global Kx_ts;
	global Y_tr;
	global Y_ts;
	global E;
	global debugging;
	global params;
	debugging = 0;
	
	% kernel file
	Kx_tr = dlmread(sprintf('%s_train', kernelfile));
	Kx_tr = Kx_tr(:,1:size(Kx_tr,1));
	Kx_ts = dlmread(sprintf('%s_test', kernelfile));
	Kx_ts = Kx_ts(:,1:size(Kx_tr,1))';

%	n_tr = size(Kx_tr,1);
%	n_ts = size(Kx_ts,1);
%	Kx = [Kx_tr;Kx_ts];
%	Kx = (Kx).^50;
%	Kx_tr = Kx(1:n_tr,1:n_tr);
%	Kx_ts = Kx(1:n_tr,n_tr+(1:n_ts));

	
	

	% target file
	Y_tr = dlmread(sprintf('%s_train', targetfile));
	Y_ts = dlmread(sprintf('%s_test', targetfile));

	% edge file
	E = dlmread(edgefile);
	
	% set parameters
	params.mlloss = 1;	% assign loss to microlabels or edges
	params.profiling = 1;	% profile (test during learning)
	params.epsilon = 1;	% stopping criterion: minimum relative duality gap
	params.C = C;		% margin slack
	params.max_CGD_iter = CGD;		% maximum number of conditional gradient iterations per example
	params.max_LBP_iter = LBP;		% number of Loopy belief propagation iterations
	params.tolerance = 1E-10;		% numbers smaller than this are treated as zero
	params.filestem = suffix;		% file name stem used for writing output
	params.profile_tm_interval = 10;	% how often to test during learning
	params.maxiter = MAXITER;		% maximum number of iterations in the outer loop
	params.verbosity = 1;
	params.debugging = 0;

	% run M3_LBP algorithm
	rtn = learn_MMCRF;
	
	% collect results
	load(sprintf('Ypred_%s.mat', suffix));
	dlmwrite(resultfile, Ypred_ts);
	
	% destroy files
	flag = system(sprintf('rm Ypred_%s.mat', suffix));
	flag = system(sprintf('rm %s.log', suffix));
	quit()



