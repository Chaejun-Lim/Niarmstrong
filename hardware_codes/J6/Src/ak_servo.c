//===================================================================================
// ak_servo.c
//
// Description:
//   Connects the Cubemars AK Series Servo Mode CAN data format to the current
//   position controller interface.
//
// Current motorCon interface:
//   Position_Control(float Degree_Ref, float RPM_max)
//
// This file does:
//   1. Decode the received 0x606 CAN frame.
//   2. Convert the received position command to AK_DegreeRef [degree].
//   3. Convert the received speed limit to AK_RPMmax [RPM].
//   4. Send the current position/speed as a 0x2906 feedback frame every 20 ms.
//
// Simplified control policy:
//   - This file only updates AK_DegreeRef and AK_RPMmax when a valid 0x606
//     command is received.
//   - If no new valid command is received, AK_DegreeRef and AK_RPMmax keep
//     their previous values.
//   - Therefore, main.c can call Position_Control(AK_DegreeRef, AK_RPMmax)
//     every control cycle without an additional validity condition.
//
// MCUViewer check:
//   RX final command:
//     - AK_Cmd_Count
//     - AK_DegreeRef [degree]
//     - AK_RPMmax    [RPM]
//
//   TX operation:
//     - AK_Tx_Count
//
// Note:
//   CAN1 initialization, RX interrupt, and raw CAN frame storage are handled
//   in can_if.c.
//===================================================================================

#include "ak_servo.h"
#include "can_if.h"
#include "motorCon.h"
#include <math.h>

//===================================================================================
// MCUViewer variables
//===================================================================================
volatile uint32_t AK_Cmd_Count = 0u;

volatile float AK_DegreeRef = 0.0f;     // [degree]
volatile float AK_RPMmax = 0.0f;        // [RPM]

volatile uint32_t AK_Tx_Count = 0u;

//===================================================================================
// AK Servo Mode RX frame processing function
//
// Target frame:
//   Extended ID = 0x606
//   Meaning:
//     (Position-Velocity Loop Mode 6 << 8) | Motor ID 6
//
// RX data format:
//   Data[0:3] = Position command
//   Data[4:5] = Speed limit
//   Data[6:7] = Acceleration limit
//
// Unit conversion:
//   Position:
//     4-byte int32
//     pos_raw = position[degree] * 10000
//     position[degree] = pos_raw / 10000
//
//   Speed:
//     2-byte int16
//     spd_raw = speed[ERPM] / 10
//     speed[ERPM] = spd_raw * 10
//
//     Because speed uses only 2 bytes, using 1 count = 1 ERPM would limit
//     the range to -32768 ~ 32767 ERPM. AK uses 1 count = 10 ERPM to express
//     a wider range within the same 2-byte field.
//
//   Acceleration:
//     2-byte int16
//     acc_raw = acceleration[ERPM/s^2] / 10
//     Currently ignored.
//
// Final values for Position_Control():
//   AK_DegreeRef [degree]
//   AK_RPMmax    [RPM]
//
// If no new CAN frame has been received:
//   This function returns immediately.
//   AK_DegreeRef and AK_RPMmax keep their previous values.
//===================================================================================
uint8_t AK_Servo_ProcessRx(uint8_t PolePair)
{
    uint32_t ext_id;
    uint8_t ide;
    uint8_t rtr;
    uint8_t dlc;
    uint8_t data[8];

    if (CAN_Rx_New == 0u)
    {
        return AK_CAN_ERROR;
    }

    //-------------------------------------------------------------------------------
    // Copy shared CAN RX data updated by the CAN RX interrupt.
    // Interrupts are briefly disabled to avoid mixed data during copying.
    //-------------------------------------------------------------------------------
    uint32_t primask = __get_PRIMASK();
    __disable_irq();

    ext_id = CAN_Rx_ExtId;
    ide    = CAN_Rx_IDE;
    rtr    = CAN_Rx_RTR;
    dlc    = CAN_Rx_DLC;

    for (uint8_t i = 0u; i < 8u; i++)
    {
        data[i] = CAN_Rx_Data[i];
    }

    CAN_Rx_New = 0u;

    __set_PRIMASK(primask);

    //-------------------------------------------------------------------------------
    // Process only AK Servo Mode extended data frames with 8 data bytes.
    //-------------------------------------------------------------------------------
    if (ide != CAN_ID_EXT)
    {
    	return AK_CAN_ERROR;
    }

    if (rtr != CAN_RTR_DATA)
    {
    	return AK_CAN_ERROR;
    }

    if (dlc != 8u)
    {
    	return AK_CAN_ERROR;
    }

    //-------------------------------------------------------------------------------
    // Process only the Position-Velocity Loop Mode command for Motor ID 6.
    // Other actuator frames are ignored.
    //-------------------------------------------------------------------------------
    if (ext_id != AK_CMD_POS_SPD_EXT_ID)
    {
    	return AK_CAN_ERROR;
    }

    //-------------------------------------------------------------------------------
    // Restore the position command.
    //
    // Data[0:3] is a big-endian int32.
    // Data[0] is the MSB, and Data[3] is the LSB.
    //-------------------------------------------------------------------------------
    uint32_t pos_u32 =
        ((uint32_t)data[0] << 24) |
        ((uint32_t)data[1] << 16) |
        ((uint32_t)data[2] << 8)  |
        ((uint32_t)data[3]);

    int32_t pos_raw = (int32_t)pos_u32;

    //-------------------------------------------------------------------------------
    // Restore the speed limit.
    //
    // Data[4:5] is a big-endian int16.
    // Data[4] is the high byte, and Data[5] is the low byte.
    //-------------------------------------------------------------------------------
    int16_t spd_raw =
        (int16_t)(((uint16_t)data[4] << 8) |
                  ((uint16_t)data[5]));

    //-------------------------------------------------------------------------------
    // Convert to the final values used by Position_Control().
    //
    // Position_Control() receives:
    //   Degree_Ref [degree]
    //   RPM_max    [RPM]
    //-------------------------------------------------------------------------------
    AK_DegreeRef = ((float)pos_raw) / 10000.0f;

    float speed_erpm = ((float)spd_raw) * 10.0f;
    AK_RPMmax = fabsf(speed_erpm) / ((float)PolePair);

    AK_Cmd_Count++;

    return AK_CAN_NORMAL;
}

//===================================================================================
// AK Servo Mode current-state feedback TX function
//
// Target frame:
//   Extended ID = 0x2906
//   Meaning:
//     (Current State Feedback 0x29 << 8) | Motor ID 6
//
// TX data format:
//   Data[0:1] = Current position
//   Data[2:3] = Current speed
//   Data[4:5] = Current
//   Data[6]   = Temperature, currently 0
//   Data[7]   = Error, currently 0
//
// TX units:
//   Position:
//     2-byte int16
//     pos_int = position[degree] * 10
//     1 count = 0.1 degree
//
//   Speed:
//     2-byte int16
//     spd_int = speed[ERPM] / 10
//     1 count = 10 ERPM
//
//   Current:
//     2-byte int16
//     cur_int = current[A] * 100
//     1 count = 0.01 A
//
// 20 ms implementation:
//   This function can be called continuously in while(1).
//   It sends only when 20 ms has elapsed based on HAL_GetTick().
//===================================================================================
void AK_Servo_SendFeedback_20ms(float Degree_M, float RPM_E, float Idc)
{
    static uint32_t tick_p = 0u;

    uint32_t tick = HAL_GetTick();

    if ((tick - tick_p) < AK_FEEDBACK_PERIOD_MS)
    {
        return;
    }

    tick_p = tick;

    CAN_TxHeaderTypeDef tx_header;
    uint8_t tx_data[8] = {0u};
    uint32_t tx_mailbox;

    //-------------------------------------------------------------------------------
    // Degree_M, RPM_E, and Idc are calculated in motorCon.c.
    // This file only reads them to build the AK feedback frame.
    //-------------------------------------------------------------------------------
    float pos_deg = Degree_M;
    float speed_erpm = RPM_E;
    float current_a = Idc;

    //-------------------------------------------------------------------------------
    // Convert to AK feedback units.
    //
    // Position:
    //   pos_int = pos_deg * 10
    //
    // Speed:
    //   spd_int = speed_erpm / 10
    //
    // Current:
    //   cur_int = current_a * 100
    //
    // The values must fit into 2-byte int16 fields.
    //-------------------------------------------------------------------------------
    float pos_tmp = MC_SAT(pos_deg * 10.0f, 32767.0f);
    float spd_tmp = MC_SAT(speed_erpm / 10.0f, 32767.0f);
    float cur_tmp = MC_SAT(current_a * 100.0f, 32767.0f);

    int16_t pos_int = (int16_t)pos_tmp;
    int16_t spd_int = (int16_t)spd_tmp;
    int16_t cur_int = (int16_t)cur_tmp;

    //-------------------------------------------------------------------------------
    // Pack int16 values as big-endian CAN data.
    //-------------------------------------------------------------------------------
    tx_data[0] = (uint8_t)((pos_int >> 8) & 0xFF);
    tx_data[1] = (uint8_t)( pos_int       & 0xFF);

    tx_data[2] = (uint8_t)((spd_int >> 8) & 0xFF);
    tx_data[3] = (uint8_t)( spd_int       & 0xFF);

    tx_data[4] = (uint8_t)((cur_int >> 8) & 0xFF);
    tx_data[5] = (uint8_t)( cur_int       & 0xFF);

    // Temperature and error are not implemented yet.
    tx_data[6] = 0u;
    tx_data[7] = 0u;

    tx_header.ExtId = AK_FB_STATE_EXT_ID;
    tx_header.IDE = CAN_ID_EXT;
    tx_header.RTR = CAN_RTR_DATA;
    tx_header.DLC = 8u;
    tx_header.TransmitGlobalTime = DISABLE;

    //-------------------------------------------------------------------------------
    // If the TX request is accepted by a mailbox, AK_Tx_Count is increased.
    // In MCUViewer, this confirms that the feedback function is operating.
    //-------------------------------------------------------------------------------
    if (HAL_CAN_GetTxMailboxesFreeLevel(&hcan1) > 0u)
    {
        if (HAL_CAN_AddTxMessage(&hcan1, &tx_header, tx_data, &tx_mailbox) == HAL_OK)
        {
            AK_Tx_Count++;
        }
    }
}

