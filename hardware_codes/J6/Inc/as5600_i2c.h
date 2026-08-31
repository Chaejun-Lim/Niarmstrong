/**
  ******************************************************************************
  * @file    as5600_i2c.h
  * @author	 Kwon Dohyeon
  * @brief   Header file of I2C for AS5600 Configuration module.
  ******************************************************************************
  */

/* Define to prevent recursive inclusion --------------------------------------*/
#ifndef AS5600_I2C_H_
#define AS5600_I2C_H_

/* Includes -------------------------------------------------------------------*/
#include "stm32f767xx.h"


/* Exported typedef -----------------------------------------------------------*/
/*  */


/* Exported define ------------------------------------------------------------*/
/* AS5600 I2C Error define */
#define AS5600_READ_ERROR      5000U

/* AS5600 Max Raw Angle define */
#define AS5600_RAW_ANGLE_MAX   4095U


/* Exported macros ------------------------------------------------------------*/
/*  */


/* Exported function prototypes -----------------------------------------------*/
/* AS5600 I2C Initialization function prototype */
void AS5600_I2C2_Init(void);

/* AS5600 Reading Angle function prototype */
uint16_t AS5600_ReadRawAngle(void);

#endif /* AS5600_I2C_H_ */
