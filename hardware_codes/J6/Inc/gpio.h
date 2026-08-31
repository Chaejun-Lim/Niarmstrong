/**
  ******************************************************************************
  * @file    gpio.h
  * @author	 Kwon Dohyeon
  * @brief   Header file of GPIO Configuration module.
  ******************************************************************************
  */

/* Define to prevent recursive inclusion --------------------------------------*/
#ifndef INC_GPIO_H_
#define INC_GPIO_H_


/* Includes -------------------------------------------------------------------*/


/* Exported typedef -----------------------------------------------------------*/
/*  */


/* Exported define ------------------------------------------------------------*/
/*  */


/* Exported macros ------------------------------------------------------------*/
/*  */


/* Exported function prototypes -----------------------------------------------*/
/* GPIO Initialization function prototype */
void Init_GPIO(void);

/* GPIO Output function prototype */
void Reset_UVW_GPIO(uint8_t U_ReSet, uint8_t V_ReSet, uint8_t W_ReSet);
void Set_UVW_GPIO(uint8_t U_Set, uint8_t V_Set, uint8_t W_Set);
/**
  * @ Set GPIO
  * GPIOE->BSRR = BIT14; // PE14(Timing Verification Using Oscilloscope) High
  * GPIOC->BSRR = BIT6;  // PC6(FLT_LED) High
  */
/**
  * @ Reset GPIO
  * GPIOE->BSRR = BIT30; // PE14(Timing Verification Using Oscilloscope) Low
  * GPIOC->BSRR = BIT22; // PC6(FLT_LED) Low
  */

#endif /* INC_GPIO_H_ */
