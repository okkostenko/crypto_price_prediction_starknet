// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.9;

import "hardhat/console.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/interfaces/IERC20.sol";
import '@uniswap/v3-periphery/contracts/interfaces/ISwapRouter.sol';

enum TradeInstractions{
    BUY,
    SELL
}

enum Tokens{
    ETH,
    USDT
}

interface IStarknetCore{
    function send_message_to_l2(
        uint256 to_address,
        uint256 selector,
        uint256[] payload
    )external returns (bytes32);

    function consume_message_from_l2(
        uint256 form_address,
        uint256[] payload
    )external returns (bytes32);
}

contract Contract is Ownable{
    ISwapRouter public immutable swap_router;
    IStarknetCore public immutable starknet_core;
    IERC20 public immutable eth;
    IERC20 public immutable usdt;

    uint256 public l2_contract_address;

    uint256 public current_amount_eth;
    uint256 public current_amount_usdt;

    uint24 public constant pool_fee = 3000;

    event receive_founds(address sender, uint amount, Tokens token);
    event execute_trade(TradeInstractions instruction, uint amount);

    constructor(ISwapRouter _swap_router, IStarknetCore _starknet_core, IERC20 _eth, IERC20 _usdt) payable {
        swap_router = _swap_router;
        starknet_core = _starknet_core;
        eth = _eth;
        usdt = _usdt;

        current_amount_eth = 0;
        current_amount_usdt = 0;
    }

    function update_l2(uint256 _l2_address) external onlyOwner{
        l2_contract_address = _l2_address;
    }

    function withdraw(Tokens token, uint amount) external onlyOwner{
        if (token == Tokens.USDT){
            usdt.transferFrom(msg.sender, address(this), amount);
            current_amount_usdt -= amount;
        }
        else if (token == Tokens.ETH) {
            eth.transferFrom(msg.sender, address(this), amount);
            current_amount_eth -= amount;
        }
    }

    function receive_instraction(TradeInstractions instraction, uint amount) external onlyOwner{
        uint256[] payload = new uint256[](2);
        payload[0] = instraction == TradeInstractions.BUY ? 0 : 1;
        payload[1] = amount;

        starknet_core.consume_message_from_l2(l2_contract_address, payload);

        if (instraction == TradeInstractions.BUY){
            buy_eth(amount);
        }
        else if (instraction == TradeInstractions.SELL){
            sell_eth(amount);
        }

        emit execute_trade(instraction, amount);
    }

    function buy_eth(uint amount) private {
        usdt.approve(address(swap_router), amount+100000);

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
        
        current_amount_usdt -= amount;
        current_amount_eth += swap_router.exactInputSingleParams(params);
    }

    function sell_eth(uint amount) private {
        eth.approve(address(swap_router), amount-100000);

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

        current_amount_eth -= amount;
        current_amount_usdt += swap_router.exactInputSingleParams(params);
    }

    function add_funds(Tokens token, uint amount) external onlyOwner{
        if (token == Tokens.ETH){
            eth.transferFrom(msg.sender, address(this), amount);
            current_amount_eth += amount;
        }
        else if (token == Tokes.USDT){
            usdt.transferFrom(msg.sender, address(this), amount);
            current_amount_usdt += amount;
        }

        emit receive_founds(msg.sender, amount, token);
    }
}