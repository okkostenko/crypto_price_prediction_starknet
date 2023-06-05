%lang starknet

// import dependencies
from starkware.cairo.common.cairo_builtins import HashBuiltin
from starkware.starknet.common.messages import send_message_to_l1
from starkware.cairo.common.math_cmp import is_le
from starknet.cairo.common.pow import pow
from starkware.starknet.common.syscalls import get_caller_address
from starkware.cairo.common.alloc import alloc
from starkware.cairo.common.registers import get_fp_and_pc
from starkware.cairo.common.serialize import serialize_word
from starkware.cairo.common.math import assert_nn_le
from starkware.cairo.common.math import signed_div_rem

// set constrants
const HOLD_STRATEGY = 0
const SELL_STRATEGY = 1
const BUY_STRATEGY = 2

const SELL_MAX = 0
const SELL = 1
const HOLD = 2
const BUY = 3
const BUY_MAX = 4

const ETH_DECIMALS = 18
const USDT_DECIMALS = 6

const EXP = 2.71828

// persistant state of owner address
@storage_var
func owner() -> (owner_address: felt):
end

// persistant state of L1 contract address
@storage_var
func l1_contract() -> (l1_contract_address: felt):
end

@storage_var
func usdt_max() -> (usdt_amount_max: felt):
end

@storage_var
func usdt() -> (usdt_amount: felt):
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
    remaining_weth: felt, remaining_usdt:felt, weth_price_ratio: felt,
    
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

    // third hidden layer parameters
    hidden_3_col_num: felt,
    hidden_3_row_num: felt,
    hidden_3_ptr_len: felt,
    hidden_3_data_ptr: felt*,
    hidden_3_bias_ptr_len: felt,
    hidden_3_bias_ptr: felt*,
    
    //forth hidden layer parameters
    hidden_4_col_num: felt,
    hidden_4_row_num: felt,
    hidden_4_ptr_len: felt,
    hidden_4_data_ptr: felt*,
    hidden_4_bias_ptr_len: felt,
    hidden_4_bias_ptr: felt*,

    //fifth hidden layer parameters
    hidden_5_col_num: felt,
    hidden_5_row_num: felt,
    hidden_5_ptr_len: felt,
    hidden_5_data_ptr: felt*,
    hidden_5_bias_ptr_len: felt,
    hidden_5_bias_ptr: felt*,

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
    let (_l1_contract_address) = l1_contract.read() // get l1 contract address
    let (msg_sender) = get_caller_address() // get caller address

    assert _owner_address = msg_sender // chaeck if owner is a caller

    // predict the price of the token
    let (output_scores_len, output_scores) = five_layer_mlp(
        input_data_ptr_len, input_data_ptr,
        hidden_1_col_num, hidden_1_row_num, hidden_1_ptr_len, hidden_1_data_ptr, hidden_1_bias_ptr_len, hidden_1_bias_ptr,
        hidden_2_col_num, hidden_2_row_num, hidden_2_ptr_len, hidden_2_data_ptr, hidden_2_bias_ptr_len, hidden_2_bias_ptr,
        hidden_3_col_num, hidden_3_row_num, hidden_3_ptr_len, hidden_3_data_ptr, hidden_3_bias_ptr_len, hidden_3_bias_ptr,
        hidden_4_col_num, hidden_4_row_num, hidden_4_ptr_len, hidden_4_data_ptr, hidden_4_bias_ptr_len, hidden_4_bias_ptr,
        hidden_5_col_num, hidden_5_row_num, hidden_5_ptr_len, hidden_5_data_ptr, hidden_5_bias_ptr_len, hidden_5_bias_ptr,
        output_col_num, output_row_num, output_ptr_len, output_data_ptr, output_bias_ptr_len, output_bias_ptr,
    )
    let (output_value) = softmax(output_scores_len, output_scores, 0, 0)
    let (max_index) = alloc()
    assert [max_index] = output_value
    let (strategy: felt*) = alloc() // allocate memory for strategy variable
    let (amount: felt*) = alloc() // allocate memory for strategy variable

    // TODO: logic of calculating the final answer
    let (usdt_amount_max) = usdt_max.read() // get max amount of usdt
    let (usdt_amount) = usdt.read() // get amount of usdt

    let weth_amount_max = usdt_amount_max * weth_price_ratio // calculate max amount of weth
    let weth_amount = usdt_amount * weth_price_ratio // calculate amount of weth

    let (overflow_usdt_max) = is_le(remaining_usdt, usdt_amount_max) // check if there is overflow of usdt
    let (overflow_usdt) = is_le(remaining_usdt, usdt_amount) // check if there is overflow of usdt

    let (overflow_weth_max) = is_le(remaining_weth, weth_amount_max) // check if there is overflow of weth
    let (overflow_weth) = is_le(remaining_weth, weth_amount) // check if there is overflow of weth

    if [max_index] == SELL_MAX:
        if overflow_weth_max == 1:
            assert [amount] = remaining_weth
        else:
            assert [amount] = weth_amount_max
        end
        assert [strategy] = SELL_STRATEGY
    end
    
    if [max_index] == SELL:
        if overflow_weth == 1:
            assert [amount] = remaining_weth
        else:
            assert [amount] = weth_amount
        end 
        assert [strategy] = SELL_STRATEGY
    end

    if [max_index] == HOLD:
        assert [amount] = 0
        assert [strategy] = HOLD_STRATEGY
    end

    if [max_index] == BUY:
        if overflow_usdt == 1:
            assert [amount] = remaining_usdt
        else:
            assert [amount] = usdt_amount
        end
        assert [strategy] = BUY_STRATEGY
    end

    if [max_index] == BUY_MAX:
        if overflow_usdt_max == 1:
            assert [amount] = remaining_usdt
        else:
            assert [amount] = usdt_amount_max
        end
        assert [strategy] = BUY_STRATEGY
    end

    let (message_payload: felt*) = alloc()
    assert message_payload[0] = [strategy]
    assert message_payload[1] = [amount]

    send_message_to_l1(to_address=_l1_contract_address, payload=message_payload, payload_size=2)
    strategy_sent_to_l2.emit(strategy=[strategy], amount=[amount])

    return (strategy=strategy, amount=amount)
end

func softmax{range_check_ptr}(scores_len: felt, scores: felt*, curr_check: felt, current_max_index: felt) -> (max_index: felt):
    if curr_check == scores_len:
        return (max_index=current_max_index)
    else:
        let (is_max) = is_le(scores[current_max_index], scores[curr_check])
        if is_max == 1:
            return softmax(scores_len, scores, curr_check+1, curr_check)
        end
        return softmax(scores_len, scores, curr_check+1, current_max_index)
    end
end

// Data Structures
struct FlattenedMatrix:
    member data: felt*,
    member col_num: felt, 
    member rw_number: felt
end

struct Vector:
    member data: felt*,
    member data_length: felt
end

// Tanh activation function
func tanh_activation_function{range_check_ptr}(x:felt) -> (tanh_x: felt):
    let(tanh_x) = (pow(EXP, x) - pow(EXP, -x)) / (pow(EXP, x) + pow(EXP, -x))
    return (tanh_x=tanh_x)
end

//// recursion over all element in vector to apply tanh function to it
func tanh_over_vector{range_check_ptr}(x_vec: Vector, tanh_vector: Vector, idx: felt):
    if idx == x_vec.data_length: // if there is no more elemnts in vector to apply Tnah function on => end of the loop
        return ()
    
    let (tanh_x) = tanh_activation_function(x_vec.data[idx]) // apply Tnah on idx'th element of the vector
    assert tanh_vector.data[idx] = tanh_x // append the element to the result vector

    tanh_over_vector(x_vec=x_vec, tanh_vector=tanh_vector, idx=idx+1) // recursevely repeat until Tnah is applied to all of the element in the vector

    return ()
end

// Tanh function wrapper
@view
func tanh{range_check_ptr}(x_vec: Vector, tanh_vec: Vector):
    
    assert x_vec.data_length = tanh_vec.data_length // check if the operation is event possible

    tanh_over_vector(x_vec=x_vec, tanh_vec=tanh_vec, idx=0) // apply Tanh to all of the element in the vector

    return ()
end


// MatMul
func matrix_row_dot_mul(
    flattend_matrix: FlattenedMatrix,
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
    flattend_matrix: FlattenedMatrix,
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
    flattend_matrix: FlattenedMatrix,
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
    nn_layer_weights: FlattenedMatrix,
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

// 5-layer MLP
//// Parameters
// x: Input data vector
// a: First matrix
// b: Second matrix
// c: Third matrix
// d: Fourth matrix
// e: Fifth matrix
// y: Output vector

// Architecture
// e @ tanh(d @ tanh(c @ tanh(b @ tanh(a @ x + a_bias) + b_bias) + c_bias) + d_bias) + e_bias
func five_layer_mlp{syscall_ptr : felt*, pedersen_ptr : HashBuiltin*, range_check_ptr}(
    x_data_ptr_len : felt, x_data_ptr : felt*,
    a_num_rows : felt, a_num_cols : felt, a_data_ptr_len : felt, a_data_ptr : felt*, a_bias_ptr_len : felt, a_bias_ptr : felt*,
    b_num_rows : felt, b_num_cols : felt, b_data_ptr_len : felt, b_data_ptr : felt*, b_bias_ptr_len : felt, b_bias_ptr : felt*,
    c_num_rows : felt, c_num_cols : felt, c_data_ptr_len : felt, c_data_ptr : felt*, c_bias_ptr_len : felt, c_bias_ptr : felt*,
    d_num_rows : felt, d_num_cols : felt, d_data_ptr_len : felt, d_data_ptr : felt*, _bias_ptr_len : felt, d_bias_ptr : felt*,
    e_num_rows : felt, e_num_cols : felt, e_data_ptr_len : felt, e_data_ptr : felt*, e_bias_ptr_len : felt, e_bias_ptr : felt*,
    y_num_rows : felt, y_num_cols : felt, y_data_ptr_len : felt, y_data_ptr : felt*, y_bias_ptr_len : felt, y_bias_ptr : felt*,
    scale_factor : felt,
) -> (output_data_ptr_len : felt, output_data_ptr : felt*):
    alloc_locals
    // Sanitycheck part 1: Matrix shapes 
    assert a_data_ptr_len = a_num_rows * a_num_cols
    assert b_data_ptr_len = b_num_rows * b_num_cols
    assert c_data_ptr_len = c_num_rows * c_num_cols
    assert d_data_ptr_len = d_num_rows * d_num_cols
    assert e_data_ptr_len = e_num_rows * e_num_cols
    assert y_data_ptr_len = y_num_rows * y_num_cols
    // Sanitycheck part 2: Dimensional analysis
    assert a_num_cols = x_data_ptr_len
    assert a_num_rows = b_num_cols
    assert b_num_rows = c_num_cols
    assert c_num_rows = d_num_cols
    assert d_num_rows = e_num_cols
    assert e_num_rows = y_num_cols
    // Construct data structures 
    local a_matrix : FlattenedMatrix = FlattenedMatrix(
        data=a_data_ptr, num_rows=a_num_rows, num_cols=a_num_cols
        )
    local a_bias : Vector = Vector(
        data_len=a_bias_ptr_len, data_ptr=a_bias_ptr
        )
    local b_matrix : FlattenedMatrix = FlattenedMatrix(
        data=b_data_ptr, num_rows=b_num_rows, num_cols=b_num_cols
        )
    local b_bias : Vector = Vector(
        data_len=b_bias_ptr_len, data_ptr=b_bias_ptr
        )
    local c_matrix : FlattenedMatrix = FlattenedMatrix(
        data=c_data_ptr, num_rows=c_num_rows, num_cols=c_num_cols
        )
    local c_bias : Vector = Vector(
        data_len=c_bias_ptr_len, data_ptr=c_bias_ptr
        )
    local d_matrix : FlattenedMatrix = FlattenedMatrix(
        data=d_data_ptr, num_rows=d_num_rows, num_cols=d_num_cols
        )
    local d_bias : Vector = Vector(
        data_len=d_bias_ptr_len, data_ptr=d_bias_ptr
        )
    local e_matrix : FlattenedMatrix = FlattenedMatrix(
        data=e_data_ptr, num_rows=e_num_rows, num_cols=e_num_cols
        )
    local e_bias : Vector = Vector(
        data_len=e_bias_ptr_len, data_ptr=e_bias_ptr
        )
    local y_matrix : FlattenedMatrix = FlattenedMatrix(
        data=y_data_ptr, num_rows=y_num_rows, num_cols=y_num_cols
        )
    local y_bias : Vector = Vector(
        data_len=y_bias_ptr_len, data_ptr=y_bias_ptr
        )

    // Construct input/intermediate data structures 
    let (x1_data_ptr : felt*) = alloc()  # After first dense layer
    let (x1_tanh_data_ptr : felt*) = alloc()  # After first tanh
    let (x2_data_ptr : felt*) = alloc()  # After second dense layer
    let (x2_tanh_data_ptr : felt*) = alloc()  # After second tanh
    let (x3_data_ptr : felt*) = alloc()  # After third dense layer
    let (x3_tanh_data_ptr : felt*) = alloc()  # After third tanh
    let (x4_data_ptr : felt*) = alloc()  # After fourth dense layer
    let (x4_tanh_data_ptr : felt*) = alloc()  # After fourth tanh
    let (x5_data_ptr : felt*) = alloc()  # After fifth dense layer
    let (x5_tanh_data_ptr : felt*) = alloc()  # After fifth tanh
    let (y_data_ptr : felt*) = alloc()  # After final dense layer

    local x : Vector = Vector(
        data_len=x_data_ptr_len, data_ptr=x_data_ptr
        )

    local x1 : Vector = Vector(
        data_len=a_num_rows, data_ptr=x1_data_ptr
        )
    local x1_tanh : Vector = Vector(
        data_len=a_num_rows, data_ptr=x1_tanh_data_ptr
        )

    local x2 : Vector = Vector(
        data_len=b_num_rows, data_ptr=x2_data_ptr
        )
    local x2_tanh : Vector = Vector(
        data_len=b_num_rows, data_ptr=x2_tanh_data_ptr
        )

    local x3 : Vector = Vector(
        data_len=c_num_rows, data_ptr=x3_data_ptr
        )
    local x3_tanh : Vector = Vector(
        data_len=c_num_rows, data_ptr=x3_tanh_data_ptr
        )

    local x4 : Vector = Vector(
        data_len=d_num_rows, data_ptr=x4_data_ptr
        )
    local x4_tanh : Vector = Vector(
        data_len=d_num_rows, data_ptr=x4_tanh_data_ptr
        )

    local x5 : Vector = Vector(
        data_len=e_num_rows, data_ptr=x5_data_ptr
        )
    local x5_tanh : Vector = Vector(
        data_len=e_num_rows, data_ptr=x5_tanh_data_ptr
        )

    local y : Vector = Vector(
        data_len=y_num_rows, data_ptr=y_data_ptr
        )

    // First layer
    linear1d_forward(x=x, a=a_matrix, a_bias=a_bias, out=x1, scale_factor=scale_factor)
    // Compute first tanh
    vector_tanh(x_vec=x1, result_vec=x1_tanh)
    // Second layer: Matmul, then add bias
    linear1d_forward(x=x1_tanh, a=b_matrix, a_bias=b_bias, out=x2, scale_factor=scale_factor)
    // Compute second tanh 
    vector_tanh(x_vec=x2, result_vec=x2_tanh)
    // Third layer: Matmul, then add bias 
    linear1d_forward(x=x2_tanh, a=c_matrix, a_bias=c_bias, out=x3, scale_factor=scale_factor)
    // Compute third tanh
    vector_tanh(x_vec=x3, result_vec=x3_tanh)
    // Fourth layer: Matmul, then add bias
    linear1d_forward(x=x3_tanh, a=d_matrix, a_bias=d_bias, out=x4, scale_factor=scale_factor)
    // Compute fourth tanh
    vector_tanh(x_vec=x4, result_vec=x4_tanh)
    // Fifth layer: Matmul, then add bias
    linear1d_forward(x=x4_tanh, a=e_matrix, a_bias=e_bias, out=x5, scale_factor=scale_factor)
    // Compute fifth tanh
    vector_tanh(x_vec=x5, result_vec=x5_tanh)
    // Output layer: Matmul, then add bias
    linear1d_forward(x=x5, a=y_matrix, a_bias=y_bias, out=y, scale_factor=scale_factor)
    // Return output

    return (output_data_ptr_len=y.data_len, output_data_ptr=y.data_ptr)
end 