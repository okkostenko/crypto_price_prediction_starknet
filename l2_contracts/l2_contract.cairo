%lang starknet

// import dependencies
from starkware.cairo.common.cairo_builtins import HashBuiltin
from starkware.starknet.common.messages import send_message_to_l1
from starkware.cairo.common.math_cmp import is_le
from starkware.starknet.common.syscalls import get_caller_address
from starkware.cairo.common.alloc import alloc
from starkware.cairo.common.registers import get_fp_and_pc
from starkware.cairo.common.serialize import serialize_word
from starkware.cairo.common.math import assert_nn_le
from starkware.cairo.common.math import signed_div_rem

// set constrants
const BUY_STRATEGY = 0
const SELL_STRATEGY = 1
const NULL_STRATEGY = 2

const BTC_DECIMALS = 18
const ETH_DECIMALS = 18
const USDT_DECIMALS = 6

// persistant state of owner address
@storage_var
func owner() -> (owner_address: felt):
end

// persistant state of L1 contract address
@storage_var
func l1_contract() -> (l1_contract_address: felt):
end

// event (sends strategy and amount to outside of the StarkNet)
@event
func strategy_sent_to_l2(strategy: felt, amount: felt):
end

// constractor (function that will be called immidiatly after a Smart Contractor is deployed to StarkNet)
@constractor
func constractor{syscalls_ptr: felt*, pedersen_ptr: HashBuiltin*, range_check_ptr: felt*}(
    _owner_address: felt, 
    _l1_contract_address: felt
):
    owner.write(value=_owner_address) // writes owner addres to the owner storage variable
    l1_contract.write(value=_l1_contract_address) // writes l1 contract addres to the l1_contract storage variable
    return ()
end

// sends calculated strategy and amount
@external // function that can be called by other contracts
func send_message{syscall_ptr: felt*, pedersen_ptr: HashBuiltin*, range_check_ptr}(
    strategy: felt,
    amount: felt
):
    let (_owner_address) = owner.read() // read the owner address from the owner storage variable
    let (_l1_contract_address) = l1_contract.read() // read the l1 contract address from the l1_contract storage variable
    let (msg_sender) = get_caller_address() // get the address of the message sender (caller)

    assert _owner_address=msg_sender // check if owner is message sender, if not - fails

    let (payload: felt*) = alloc() // allocate memory for a message payload 
    assert payload[0] = strategy // set first element of the payload to strategy
    assert payload[1] = amount // set second element of the payload to amount

    // send message to L1 (built in function)
    send_message_to_l1(
        to_address = _l1_contract_address, // to address of the L1 contract
        payload_size = 2, // payload size is 2 (strategy, amount)
        payload = payload, // payload
    )

    // store strategy and amount in transaction log (can be accessible using contract address)
    strategy_sent_to_l2.emit(strategy=strategy, amount=amount)

    return ()
end

// calculates strategy and amount using NN
@external
func calculate_strategy{syscall_ptr: felt*, pedersen_ptr: HashBuiltin*, range_check_ptr}(
    remaining_eth: felt, remaining_usdt:felt, eth_price_ratio: felt,
    
    // input data parameters
    input_data_ptr_len: felt,
    input_data_ptr: felt*,

    // first hidden layer parameters
    hidden_1_col_num: felt,
    hidden_1_row_num: felt,
    hidden_1_ptr_len: felt,
    hidden_1_data_ptr: felt*,
    hidden_1_bias_ptr_len: felt,
    hidden_1_bias_ptr: felt*,

    // second hidden layer parameters
    hidden_2_col_num: felt,
    hidden_2_row_num: felt,
    hidden_2_ptr_len: felt,
    hidden_2_data_ptr: felt*,
    hidden_2_bias_ptr_len: felt,
    hidden_2_bias_ptr: felt*,

    // output layer parameters
    output_col_num: felt,
    output_row_num: felt,
    output_ptr_len: felt,
    output_data_ptr: felt*,
    output_bias_ptr_len: felt,
    output_bias_ptr: felt*,

    //scale factor
    scale_factor: felt
) -> (strategy: felt, amount: felt): // returns strategy and amount
    alloc_locals

    let (_owner_address) = owner.read() // get owner address
    let(_l1_contract_address) = l1_contract.read() // get l1 contract address
    let(msg_sender) = get_caller_address() // get caller address

    assert _owner_address = msg_sender // chaeck if owner is a caller

    // predict the price of the token
    let (price_prediction) = nn_model_predict(
        input_data_ptr_len, input_data_ptr,
        hidden_1_col_num, hidden_1_row_num, hidden_1_ptr_len, hidden_1_data_ptr, hidden_1_bias_ptr_len, hidden_1_bias_ptr,
        hidden_2_col_num, hidden_2_row_num, hidden_2_ptr_len, hidden_2_data_ptr, hidden_2_bias_ptr_len, hidden_2_bias_ptr,
        output_col_num, output_row_num, output_ptr_len, output_data_ptr, output_bias_ptr_len, output_bias_ptr,
    )

    let (strategy: felt*) = alloc() // allocate memory for strategy variable
    let (amount: felt*) = alloc() // allocate memory for strategy variable

    // TODO: logic of calculating thi final answer

    strategy_sent_to_l2.emit(strategy=[strategy], amount=[amount])

    return (strategy=strategy, amount=amount)
end

// Data Structures
struct FlattendMatrix:
    member data: felt*,
    member col_num: felt, 
    member rw_number: felt
end

struct Vector:
    member data: felt*,
    member data_length: felt
end

// Relu activation function
func relu_activation_function{range_check_ptr}(x:felt) -> (relu_x: felt):
    let (multiplier) = is_le(0, x) // If x > 0 => multiplier = 1, otherwise, multiplier = 0
    let (relu_x) = multiplier * x // relu_x = max(0, x)

    return (relu_x=relu_x)
end

//// recursion over all element in vector to apply relu function to it
func relu_over_vector{range_check_ptr}(x_vec: Vector, relu_vector: Vector, idx: felt):
    i
    f idx == x_vec.data_length: // if there is no more elemnts in vector to apply ReLU function on => end of the loop
        return ()
    
    let (relu_x) = relu_activation_function(x_vec.data[idx]) // apply ReLU on idx'th element of the vector
    assert relu_vector.data[idx] = relu_x // append the element to the result vector

    relu_over_vector(x_vec=x_vec, relu_vector=relu_vector, idx=idx+1) // recursevely repeat until ReLU is applied to all of the element in the vector

    return ()
end

// Relu function wrapper
@view
func relu{range_check_ptr}(x_vec: Vector, relu_vec: Vector):
    
    assert x_vec.data_length = relu_vec.data_length // check if the operation is event possible

    relu_over_vector(x_vec=x_vec, relu_vec=relu_vec, idx=0) // apply ReLU to all of the element in the vector

    return ()
end


// MatMul
func matrix_row_dot_mul(
    flattend_matrix: FlattendMatrix,
    vector: Vector,
    row_idx: felt,
    col_idx: felt
) -> (result: felt):
    
    alloc_locals

    if col_idx == flattend_matrix.col_num: // end of the loop
        return (0)
    end

    local matrix_weight = flattend_matrix.data[flattend_matrix.col_num * row_idx + col_idx] // get matrix element
    local result = matrix_weight * vector.data[col_idx] // multiply the element by the corresponding element of the vector
    
    let (rest) = matrix_row_dot_mul(
        flattend_matrix=flattend_matrix, 
        vector=vector, 
        row_idx=row_idx, 
        col_idx=col_idx+1) // repeat for all elements in the row

    return (result + rest)
end

func matmul_by_row(
    flattend_matrix: FlattendMatrix,
    vector: Vector,
    result_vector: Vector,
    row_idx: felt
):
    if row_idx == flattend_matrix.row_num: // end of the loop
        return ()
    end

    let (row_result) = matrix_row_dot_mul(
        flattend_matrix=flattend_matrix, 
        vector=vector, 
        row_idx=row_idx, 
        col_idx=0
    ) // compute matrix multiplication for the current row for each element, starting for 0th

    assert result_vector.data[row_idx]  = row_result // append the result to the result vector
    matmul_by_row(
        flattend_matrix=flattend_matrix,
        vector=vector,
        result_vector=result_vector,
        row_idx=row_idx+1
    ) // repeat for each row of the matrix

    return ()
end

func matmul(
    flattend_matrix: FlattendMatrix,
    vector: Vector,
    result_vector: Vector
):
    matmul_by_row(
        flattend_matrix=flattend_matrix, 
        vector=vector,
        result_vector=vector,
        row_idx=0
    ) // compute matrix multiplication

    return ()
end

// Vector Addition

func vec_add{range_check_ptr}(vector_1: Vector, vector_2:Vector, result_vector: Vector):
    
    assert vector_1.data_length = vector_2.data_length // check if the vectors are the same length

    if vector_1.data_length == 0: // if there is no elements in the vectors, end the loop
        return ()
    end

    vec_add_by_element(vector_1=vector_1, vector_2=vector_2, idx=0) // add vectors element by element
    return ()
end

func vec_add_by_element{range_check_ptr}(
    vector_1: Vector, 
    vector_2: Vector, 
    result: Vector, 
    idx: felt
):

    if idx == result.data_length: // end the loop
        return ()
    end

    assert result.data[idx] = vector_1.data[idx] + vector_2.data[idx] // append the sum of idx'th elements of each vector

    vec_add_by_element(vector_1=vector_1, vector_2=vector_2, result=result, idx=idx+1) // repeat until all corresponding elements of each vector are added together 

    return ()
end

// Scale Vector

func scale_vec_by_element{range_check_ptr}(
    vector: Vector,
    scaled_vector: Vector,
    scale_factor: felt,
    idx: felt
):
    if idx == vector.data_length: // end the loop
        return ()
    end

    let (coefficient: felt, remainder: felt) = signed_div_rem(
        value=vector.data[idx], div=scale_factor, bound=1000000000000000000000000
    ) // devide current element of the vector by scale_factor

    assert scaled_vector[idx] = coefficient // append the integer part of the division to the result vector
    scale_vec_by_element(
        vector=vector,
        scaled_vector=scaled_vector,
        scale_factor=scale_factor,
        idx=idx+1
    ) // repeat for all eleements in the vector

    return ()
end

// Neura Net

//// forward computation of the dense layer of the NN
func dense_layer{range_check_ptr}(
    input: Vector,
    nn_layer_weights: FlattendMatrix,
    nn_layer_bias: Vector,
    output: Vector,
    scale_factor:felt
):

    alloc_locals

    // check the dimentions of the layers
    assert nn_layer_weights.col_num = input.data_length 
    assert nn_layer_weights.row_num = output.data_length
    assert nn_layer_weights.row_num = nn_layer_bias.data_length

    let (post_weights: felt*) = alloc() // create a variable for the result
    local post_weights_vector: Vector = Vector(data_length=nn_layer_weights.row_num, data=post_weights) // fill the result vector
    let (scaled_post_weight: felt*) = alloc()
    local scaled_post_weight_vector: Vector = Vector(data_length=nn_layer_weights.row_num, data=scaled_post_weights) // scale the vector

    matmul(vector=post_weights_vector, scaled_vector=scaled_post_weight_vector, scale_factor=scale_factor) // compute matrix multiplication

    vec_add(vector_1=scaled_post_weight_vector, vector_2=nn_layer_bias, result=out) // add the bias the the result vector
    
    return ()
end

    