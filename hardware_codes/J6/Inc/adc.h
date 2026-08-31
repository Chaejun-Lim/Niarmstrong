/**
  ******************************************************************************
  * @file    adc.h
  * @author	 Kwon Dohyeon
  * @brief   Header file of ADC Configuration module.
  ******************************************************************************
  */

/* Define to prevent recursive inclusion --------------------------------------*/
#ifndef INC_ADC_H_
#define INC_ADC_H_


/* Includes -------------------------------------------------------------------*/


/* Exported typedef -----------------------------------------------------------*/
/*  */


/* Exported define ------------------------------------------------------------*/
/* ADC Specification defines */
#define ADC_VREF 3.3f
#define ADC_RESOLUTION 4095U


/* Exported macros ------------------------------------------------------------*/
/*  */


/* Exported function prototypes -----------------------------------------------*/
/* ADC Initialization function prototype */
void Init_ADC(void);

/* ADC Conversion function prototype */
/*
 * ADC Conversion
 * ADC1->CR2 |= BIT30; 	  	   // ADC 변환 시작
 * while((ADC1->SR & BIT1) == 0); // ADC 변환 완료까지 대기
 * (ADC1->DR); // ADC 변환 결과
 */

#endif /* INC_ADC_H_ */
