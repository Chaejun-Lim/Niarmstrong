#ifndef AK_SERVO_H_
#define AK_SERVO_H_

//===================================================================================
// ak_servo.h
//
// Description:
//   Connects the Cubemars AK Series Servo Mode CAN data format to the current
//   position controller interface.
//
// Current motorCon interface:
//   Position_Control(float Degree_Ref, float RPM_max)
//
// Current implementation:
//   1. Receive only the Position-Velocity Loop Mode command for Motor ID 6.
//      - RX Extended ID = 0x606
//      - RX position command -> AK_DegreeRef [degree]
//      - RX speed limit      -> AK_RPMmax [RPM]
//
//   2. Send current position/speed feedback every 20 ms.
//      - TX Extended ID = 0x2906
//      - Position and speed are filled.
//      - Current, temperature, and error are sent as 0.
//
// Simplified control policy:
//   - Position_Control() always uses AK_DegreeRef and AK_RPMmax.
//   - If no new CAN command is received, the previous values are kept.
//   - Before the first CAN command, AK_DegreeRef = 0 deg and AK_RPMmax = 0 RPM.
//
// MCUViewer variables to check:
//   AK_Cmd_Count
//   AK_DegreeRef
//   AK_RPMmax
//   AK_Tx_Count
//
// Note:
//   CAN1 initialization and raw CAN frame reception are handled in can_if.c.
//===================================================================================

#include "main.h"

//===================================================================================
// Cubemars AK Series Servo Mode settings
//===================================================================================
#define AK_MOTOR_ID                 6u

#define AK_CMD_POS_SPD              6u
#define AK_FB_CURRENT_STATE         0x29u

#define AK_CMD_POS_SPD_EXT_ID       (((uint32_t)AK_CMD_POS_SPD << 8) | AK_MOTOR_ID)
#define AK_FB_STATE_EXT_ID          (((uint32_t)AK_FB_CURRENT_STATE << 8) | AK_MOTOR_ID)

#define AK_FEEDBACK_PERIOD_MS       20u   // 20 ms = 50 Hz

#define AK_CAN_ERROR 				0U
#define AK_CAN_NORMAL 				1U

//===================================================================================
// MCUViewer variables
//===================================================================================
extern volatile uint32_t AK_Cmd_Count;    // Number of valid 0x606 commands

extern volatile float AK_DegreeRef;       // [degree] Position_Control() position reference
extern volatile float AK_RPMmax;          // [RPM]    Position_Control() speed limit

extern volatile uint32_t AK_Tx_Count;     // Number of successful 0x2906 feedback requests

//===================================================================================
// AK Servo Mode functions
//===================================================================================
uint8_t AK_Servo_ProcessRx(uint8_t PolePair);
void AK_Servo_SendFeedback_20ms(float Degree_M, float RPM_E, float Idc);

#endif /* AK_SERVO_H_ */
