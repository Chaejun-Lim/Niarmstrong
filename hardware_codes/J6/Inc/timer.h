/**
  ******************************************************************************
  * @file    timer.h
  * @author	 Kwon Dohyeon
  * @brief   Header file of Timer Configuration module.
  ******************************************************************************
  */

/* Define to prevent recursive inclusion --------------------------------------*/
#ifndef INC_TIMER_H_
#define INC_TIMER_H_


/* Includes -------------------------------------------------------------------*/


/* Exported typedef -----------------------------------------------------------*/
/*  */


/* Exported define ------------------------------------------------------------*/
/*  */


/* Exported macros ------------------------------------------------------------*/
/*  */


/* Exported function prototypes -----------------------------------------------*/
/* Timer Initialization function prototypes */
void Init_UVW_PWM(void);
void Init_Timer_Intterupt_for_Control(uint16_t MicroSec, uint32_t Priority);
void Init_Timer_Intterupt_for_Timeout(uint32_t MilliSec, uint32_t Priority);
void Init_Microsec_Timer(void);
void Init_Delay_MilliSec(void);

/* Delay function prototype */
void Delay_MilliSec(uint32_t MilliSec); // [ms]

/* PWM function prototypes */
void Disable_UVW_PWM(uint8_t U_DIS, uint8_t V_DIS, uint8_t W_DIS);
void Enable_UVW_PWM(uint8_t U_EN, uint8_t V_EN, uint8_t W_EN, float U_Duty_Ratio, float V_Duty_Ratio, float W_Duty_Ratio);

/* Timeout function prototype */
void Start_Timeout(void);
void Stop_Timeout(void);

#endif /* INC_TIMER_H_ */
