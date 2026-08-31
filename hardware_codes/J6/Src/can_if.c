
#include "can_if.h"

CAN_HandleTypeDef hcan1;

volatile uint32_t CAN_Rx_Count = 0u;
volatile uint32_t CAN_Rx_ExtId = 0u;
volatile uint32_t CAN_Rx_StdId = 0u;
volatile uint8_t CAN_Rx_IDE = 0u;
volatile uint8_t CAN_Rx_RTR = 0u;
volatile uint8_t CAN_Rx_DLC = 0u;
volatile uint8_t CAN_Rx_Data[8] = {0u};
volatile uint8_t CAN_Rx_New = 0u;

void CAN1_IF_Init(void)
{
    CAN_FilterTypeDef filter;

    hcan1.Instance = CAN1;
    hcan1.Init.Prescaler = 3;
    hcan1.Init.Mode = CAN_MODE_NORMAL;
    hcan1.Init.SyncJumpWidth = CAN_SJW_1TQ;
    hcan1.Init.TimeSeg1 = CAN_BS1_13TQ;
    hcan1.Init.TimeSeg2 = CAN_BS2_4TQ;
    hcan1.Init.TimeTriggeredMode = DISABLE;
    hcan1.Init.AutoBusOff = ENABLE;
    hcan1.Init.AutoWakeUp = DISABLE;
    hcan1.Init.AutoRetransmission = ENABLE;
    hcan1.Init.ReceiveFifoLocked = DISABLE;
    hcan1.Init.TransmitFifoPriority = DISABLE;

    if (HAL_CAN_Init(&hcan1) != HAL_OK)
    {
        Error_Handler();
    }

    // 우선 모든 CAN frame 수신 허용
    filter.FilterBank = 0;
    filter.FilterMode = CAN_FILTERMODE_IDMASK;
    filter.FilterScale = CAN_FILTERSCALE_32BIT;
    filter.FilterIdHigh = 0x0000;
    filter.FilterIdLow = 0x0000;
    filter.FilterMaskIdHigh = 0x0000;
    filter.FilterMaskIdLow = 0x0000;
    filter.FilterFIFOAssignment = CAN_RX_FIFO0;
    filter.FilterActivation = ENABLE;
    filter.SlaveStartFilterBank = 14;

    if (HAL_CAN_ConfigFilter(&hcan1, &filter) != HAL_OK)
    {
        Error_Handler();
    }

    if (HAL_CAN_Start(&hcan1) != HAL_OK)
    {
        Error_Handler();
    }

    if (HAL_CAN_ActivateNotification(&hcan1, CAN_IT_RX_FIFO0_MSG_PENDING) != HAL_OK)
    {
        Error_Handler();
    }
}

void HAL_CAN_RxFifo0MsgPendingCallback(CAN_HandleTypeDef *hcan)
{
    CAN_RxHeaderTypeDef rx_header;
    uint8_t rx_data[8] = {0u};

    if (hcan->Instance != CAN1)
    {
        return;
    }

    if (HAL_CAN_GetRxMessage(hcan, CAN_RX_FIFO0, &rx_header, rx_data) != HAL_OK)
    {
        return;
    }

    CAN_Rx_ExtId = rx_header.ExtId;
    CAN_Rx_StdId = rx_header.StdId;
    CAN_Rx_IDE = (uint8_t)rx_header.IDE;
    CAN_Rx_RTR = (uint8_t)rx_header.RTR;
    CAN_Rx_DLC = (uint8_t)rx_header.DLC;

    for (uint8_t i = 0u; i < 8u; i++)
    {
        CAN_Rx_Data[i] = rx_data[i];
    }

    CAN_Rx_Count++;
    CAN_Rx_New = 1u;
}
