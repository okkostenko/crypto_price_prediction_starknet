// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.9;

import "hardhat/console.sol"; 
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/interfaces/IERC20.sol";
import '@uniswap/v3-periphery/contracts/interfaces/ISwapRouter.sol';

// Set the trade instructions and tokens
enum TradeInstractions{
    BUY,
    SELL
}

enum Tokens{
    ETH,
    USDT
}

// Interface for the Starknet Core contract
interface IStarknetCore{
    function send_message_to_l2( // Send a message to the L2 contract
        uint256 to_address,
        uint256 selector,
        uint256[] payload
    )external returns (bytes32);

    function consume_message_from_l2( // Consume a message from the L2 contract
        uint256 form_address,
        uint256[] payload
    )external returns (bytes32);
}

// The main contract
contract Contract is Ownable{
    ISwapRouter public immutable swap_router; // The Uniswap V3 SwapRouter contract
    IStarknetCore public immutable starknet_core; // The Starknet Core contract
    IERC20 public immutable eth; // The ETH token
    IERC20 public immutable usdt; // The USDT token

    uint256 public l2_contract_address; // The L2 contract address

    uint256 public current_amount_eth; // The current amount of ETH in the contract
    uint256 public current_amount_usdt; // The current amount of USDT in the contract

    uint24 public constant pool_fee = 3000; // The pool fee

    event receive_founds(address sender, uint amount, Tokens token); // Event for receiving funds
    event execute_trade(TradeInstractions instruction, uint amount); // Event for executing a trade

    // The constructor
    constructor(ISwapRouter _swap_router, IStarknetCore _starknet_core, IERC20 _eth, IERC20 _usdt) payable {
        swap_router = _swap_router;
        starknet_core = _starknet_core;
        eth = _eth;
        usdt = _usdt;

        current_amount_eth = 0;
        current_amount_usdt = 0;
    }

    // The function to update the L2 contract address
    function update_l2(uint256 _l2_address) external onlyOwner{
        l2_contract_address = _l2_address; 
    }

    // The function to withdraw funds
    function withdraw(Tokens token, uint amount) external onlyOwner{
        // Choose to token to withdraw
        if (token == Tokens.USDT){ 
            usdt.transferFrom(msg.sender, address(this), amount); // Transfer the funds
            current_amount_usdt -= amount;
        }
        else if (token == Tokens.ETH) {
            eth.transferFrom(msg.sender, address(this), amount); // Transfer the funds
            current_amount_eth -= amount;
        }
    }

    // The function to receive instructions from the L2 contract
    function receive_instraction(TradeInstractions instraction, uint amount) external onlyOwner{
        uint256[] payload = new uint256[](2); // Create the payload
        payload[0] = instraction == TradeInstractions.BUY ? 0 : 1; // Set the instruction to 0 if it is a buy and 1 if it is a sell
        payload[1] = amount; // Set the amount

        starknet_core.consume_message_from_l2(l2_contract_address, payload); // Consume the message from the L2 contract

        if (instraction == TradeInstractions.BUY){ // Check the instruction
            buy_eth(amount); // Execute the trade
        }
        else if (instraction == TradeInstractions.SELL){
            sell_eth(amount); // Execute the trade
        }

        emit execute_trade(instraction, amount); // Emit the event
    }

    // The function to buy ETH
    function buy_eth(uint amount) private {
        usdt.approve(address(swap_router), amount+100000); // Approve the USDT token

        // Set the trade parameters
        ISwapRouter.ExactInputSingleParams memory params = 
            ISwapRouter.ExactInputSingleParams({
                token_in: address(usdt),
                token_out: address(eth),
                fee: pool_fee,
                recipient: address(this),
                deadline: (block.timestamp + 60*500),
                amount_in: amount,
                min_amount_out: 0,
                sqrt_price_limit_x96: 0
            });
        
        current_amount_usdt -= amount; // Update the current amount of USDT
        current_amount_eth += swap_router.exactInputSingleParams(params); // Update the current amount of ETH
    }   

    // The function to sell ETH
    function sell_eth(uint amount) private {
        eth.approve(address(swap_router), amount-100000); // Approve the ETH token

        // Set the trade parameters
        ISwapRouter.ExactInputSingleParams memory params = 
            ISwapRouter.ExactInputSingleParams({
                token_in: address(eth),
                token_out: address(usdt),
                fee: pool_fee,
                recipient: address(this),
                deadline: (block.timestamp + 60*500),
                amount_in: amount,
                min_amount_out: 0,
                sqrt_price_limit_x96: 0
            });

        current_amount_eth -= amount; // Update the current amount of ETH
        current_amount_usdt += swap_router.exactInputSingleParams(params); // Update the current amount of USDT
    }

    // The function to add funds
    function add_funds(Tokens token, uint amount) external onlyOwner{
        if (token == Tokens.ETH){
            eth.transferFrom(msg.sender, address(this), amount);
            current_amount_eth += amount;
        }
        else if (token == Tokes.USDT){
            usdt.transferFrom(msg.sender, address(this), amount);
            current_amount_usdt += amount;
        }

        emit receive_founds(msg.sender, amount, token); // Emit the event
    }
}