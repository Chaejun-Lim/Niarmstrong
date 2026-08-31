/**
  ******************************************************************************
  * @file    motorCon.h
  * @author	 Kwon Dohyeon
  * @brief   Header file of BLDC Motor Control module.
  ******************************************************************************
  */

/* Define to prevent recursive inclusion --------------------------------------*/
#ifndef INC_MOTORCON_H_
#define INC_MOTORCON_H_


/* Includes -------------------------------------------------------------------*/
#include "stm32f767xx.h"
#include "control_block.h"


/* Exported typedef -----------------------------------------------------------*/
/* BLDC handle Structure definition */
typedef struct __BLDC_HandleTypeDef
{
	/* Configuration */
	float Calibration_Offset;   	// Calibration Offset [degree]
	float Align_Offset;         	// Align Offset [degree]
	uint8_t PolePair;      	    	// Pole Pair
    float V_Limit;			    	// Voltage Limit [V]
    float V_Inverter;               // Inverter Voltage [V]

	/* States */
	float Degree_M;					// Mechanical Angle in range -180 to 180 [degree]
	float Theta;					// Mechanical Angle [rad]
    float W_M;       		    	// Mechanical Angle Velocity [rad/s]
    float RPM_M;       		    	// Mechanical Angle Velocity [rpm]
    float RPM_E;       		    	// Electrical Angle Velocity [erpm]
    uint8_t Sector;					// Sector of BLDC Motor
    uint8_t Sector_p;				// Previous Sector
    uint8_t rule;					// Current Selection Rule
    float Ias;						// Current of Phase A [A]
    float Ibs;						// Current of Phase A [A]
    float Ics;						// Current of Phase A [A]
    float Idc;						// Equivalent DC Current [A]
    uint8_t OcCnt;					// Over Current Count
    uint8_t ErrCode;				// Error Code
} BLDC_HandleTypeDef;


/* Exported define ------------------------------------------------------------*/
/*  */


/* Exported macros ------------------------------------------------------------*/
/*  */


/* Exported function prototypes -----------------------------------------------*/
/* BLDC Motor Control function prototypes */
void BLDC_Init_State(BLDC_HandleTypeDef *hbldc);
void BLDC_Align(void);
void BLDC_PowerOff(void);
void BLDC_Calc_Pos_N180_P180(BLDC_HandleTypeDef *hbldc, float Degree_M_Abs);
void BLDC_Calc_Sector(BLDC_HandleTypeDef *hbldc, float Degree_M_Abs);
void BLDC_Calc_DC_Current(BLDC_HandleTypeDef *hbldc);
void BLDC_6Step_Commutation_CCW(BLDC_HandleTypeDef *hbldc, float VRef);
void BLDC_6Step_Commutation_CW(BLDC_HandleTypeDef *hbldc, float VRef);

#endif /* INC_MOTORCON_H_ */
