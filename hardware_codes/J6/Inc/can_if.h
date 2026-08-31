
#ifndef CAN_IF_H_
#define CAN_IF_H_

#include "main.h"

extern CAN_HandleTypeDef hcan1;

extern volatile uint32_t CAN_Rx_Count;
extern volatile uint32_t CAN_Rx_ExtId;
extern volatile uint32_t CAN_Rx_StdId;
extern volatile uint8_t CAN_Rx_IDE;
extern volatile uint8_t CAN_Rx_RTR;
extern volatile uint8_t CAN_Rx_DLC;
extern volatile uint8_t CAN_Rx_Data[8];
extern volatile uint8_t CAN_Rx_New;

void CAN1_IF_Init(void);

#endif
