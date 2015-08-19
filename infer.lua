
--[[

This file trains a character-level multi-layer RNN on text data

Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6
but modified to have multi-layer support, GPU support, as well as
many other common model/optimization bells and whistles.
The practical6 code is in turn based on 
https://github.com/wojciechz/learning_to_execute
which is turn based on other stuff in Torch, etc... (long lineage)

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'
local CharSplitLMMinibatchLoader = require 'util.CharSplitLMMinibatchLoader'
local model_utils = require 'util.model_utils'
local LSTM = require 'model.LSTM'
local GRU = require 'model.GRU'
local RNN = require 'model.RNN'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:text()
cmd:text('Options')
cmd:option('-model','model to load')
cmd:option('-seq_length',50,'number of timesteps to unroll for')
cmd:option('-data_dir','','input data. default is to use what the model was trained on')
cmd:option('-output','','where to write the inferred values. If not specified, do compute them')
cmd:option('-batch_size',50,'number of sequences to test on in parallel')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-byte','0','whether to use byte encoding for input data')
cmd:option('-numblocks','100','how many blocks to process')

cmd:text()

function gprint(str)
    if opt.verbose == 1 then print(str) end
end

-- parse input params
opt = cmd:parse(arg)

-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end




-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
    gprint('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)
protos = checkpoint.protos
protos.rnn:evaluate() -- put in eval mode so that dropout works properly

-- create the data loader class
local split_sizes = {1.0,0,0}
local loader 
if(opt.byte == 1) then
    loader = CharSplitLMMinibatchLoaderByte.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes)
else
    loader = CharSplitLMMinibatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes)
end

local vocab_size = loader.vocab_size  -- the number of distinct characters
local vocab = loader.vocab_mapping
print('vocab size: ' .. vocab_size)

-- initialize the vocabulary (and its inverted version)
local vocab = checkpoint.vocab
local ivocab = {}
for c,i in pairs(vocab) do ivocab[i] = c end

-- initialize the rnn state to all zeros
gprint('creating an ' .. checkpoint.opt.model .. '...')
local current_state
current_state = {}
for L = 1,checkpoint.opt.num_layers do
    -- c and h for all layers
    local h_init = torch.zeros(1, checkpoint.opt.rnn_size):double()
    if opt.gpuid >= 0 then h_init = h_init:cuda() end
    table.insert(current_state, h_init:clone())
    if checkpoint.opt.model == 'lstm' then
        table.insert(current_state, h_init:clone())
    end
end
state_size = #current_state
-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
    gprint('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)
protos = checkpoint.protos
protos.rnn:evaluate() -- put in eval mode so that dropout works properly

-- initialize the vocabulary (and its inverted version)
local vocab = checkpoint.vocab
local ivocab = {}
for c,i in pairs(vocab) do ivocab[i] = c end

-- initialize the rnn state to all zeros
gprint('creating an ' .. checkpoint.opt.model .. '...')
local current_state
current_state = {}
for L = 1,checkpoint.opt.num_layers do
    -- c and h for all layers
    local h_init = torch.zeros(1, checkpoint.opt.rnn_size):double()
    if opt.gpuid >= 0 then h_init = h_init:cuda() end
    table.insert(current_state, h_init:clone())
    if checkpoint.opt.model == 'lstm' then
        table.insert(current_state, h_init:clone())
    end
end
state_size = #current_state



-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end


-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)

-- initialization
if do_random_init then
params:uniform(-0.08, 0.08) -- small numbers uniform
end

print('number of parameters in the model: ' .. params:nElement())
-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

init_state = {}
for L=1,checkpoint.opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, checkpoint.opt.rnn_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone())
    if checkpoint.opt.model == 'lstm' then
        table.insert(init_state, h_init:clone())
    end
end


-- do fwd/bwd and return loss, grad_params
local init_state_global = clone_list(init_state)

function load_next()
    local x, y = loader:next_batch(1)
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
       -- y = y:float():cuda()
    end
    return x
end


function infer(x)
        ------------------- forward pass -------------------
    local rnn_state = {[0] = init_state_global}
    local predictions = {}           -- softmax outputs
    local loss = 0
    for t=1,opt.seq_length do
        local lst = clones.rnn[t]:forward{x[{{}, t}], unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
        predictions[t] = lst[#lst] -- last element is the prediction
    end
    
    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
    
end

local function findNode(module,name)
    local node
    local count = 0
    for i,forwardNode in ipairs(module.forwardnodes) do
        if(forwardNode.data.annotations.name == name) then
            node =  forwardNode
            count = count + 1
        end
   end
    assert(count == 1,'did not find it the right number of times. count = '..count)

   return node
end

local function getOutput(node)
    return node.data.module.output
end
clones.forget_gates = {}
clones.state = {}
for i = 1,#clones.rnn do
    table.insert(clones.forget_gates,getOutput(findNode(clones.rnn[i],'update')))
--    table.insert(clones.state,getOutput(findNode(clones.rnn[i],'out')))
end

local splitter = nn.SplitTable(1,1)
local function rowVectorStr(t)
    return table.concat(splitter:forward(t),' ')
end

local total = 0
local cnt = 0

local computeActivations = opt.output ~= nil
local outfile 
if(computeActivations) then outfile = io.open(opt.output, "w") end

for trial = 1,opt.numblocks do
   local x = load_next()
   infer(x)

   for timestep = 1,x:size(2) do
        local gate_output = clones.forget_gates[timestep]:clone()
        total = total + gate_output:add(-0.5):mul(2):abs():mean()
        cnt = cnt  + 1

        for exampleIdx = 1,x:size(1) do
            local wordIdx = x[exampleIdx][timestep]
            local word = ivocab[wordIdx]
            if(computeActivations) then
                local activationStr = rowVectorStr(gate_output[exampleIdx])
                outfile:write(string.format('%s %s\n',word,activationStr))
            end
        end
    end
    
end

local avg = total/cnt
    print(avg)


