/**
  ******************************************************************************
  * @file    control_block.h
  * @author	 Kwon Dohyeon
  * @brief   Header file of Control block module.
  ******************************************************************************
  */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef INC_CONTROL_BLOCK_H_
#define INC_CONTROL_BLOCK_H_


/* Includes ------------------------------------------------------------------*/
#include <stdint.h>


/* Exported typedef -----------------------------------------------------------*/
/* P Controller handle Structure definition */
typedef struct __P_HandleTypeDef
{
	/* Configuration */
    float Kp;         	   // Proportional Gain
    float OutMax;     	   // Output Magnitude Limit
} P_HandleTypeDef;

/* PI Controller handle Structure definition */
typedef struct __PI_HandleTypeDef
{
	/* Configuration */
    float Kp;         	   // Proportional Gain
    float Ki;         	   // Integral Gain
    float Ka;         	   // Anti-Windup Gain
    float OutMax;     	   // Output Magnitude Limit

    /* State */
    uint8_t Is_First_Time; // First Time Flag
    float Intg;       	   // Integrator State
    uint32_t Time_In_p;	   // Previous Time Input [us]
} PI_HandleTypeDef;

/* 1st-order LPF(Bilinear Transformation) handle Structure definition */
typedef struct __LPF_HandleTypeDef
{
	/* Configuration */
	float Fc;	      	   // LPF Cut Off Frequency [Hz]
	uint32_t Tsamp;        // Sampling Time [us]
    float K1, K2;     	   // LPF Gain

    /* State */
    float Xk;         	   // LPF State
} LPF_HandleTypeDef;

/* Multi-Step Backward Difference handle Structure definition */
typedef struct __MSBD_HandleTypeDef
{
	/* Configuration */
	uint8_t Step_Num;  	   // Step Number, maximum = 30

	/* State */
	uint8_t Index;    	   // Present Index
	float Buf[31];    	   // Input Buffer
	uint32_t Time_Buf[31]; // Time Input Buffer [us]
} MSBD_HandleTypeDef;


/* Exported define ------------------------------------------------------------*/
/* Pi define */
#define PI 3.14159265f


/* Exported macros ------------------------------------------------------------*/
/* Saturation macro */
#define MC_SAT(x, limit) ((x) > (limit) ? (limit) : ((x) < -(limit) ? -(limit) : (x)))


/* Exported function prototypes -----------------------------------------------*/
/* P Controller function prototypes */
float P_Calc(P_HandleTypeDef *hp, float Ref, float Fdb);

/* PI Controller function prototypes */
void PI_Init_State(PI_HandleTypeDef *hpi);
float PI_Calc(PI_HandleTypeDef *hpi, float Ref, float Fdb, uint32_t Time_In);

/* LPF function prototypes */
void LPF_Init_State(LPF_HandleTypeDef *hlpf);
void LPF_Calc_Gain(LPF_HandleTypeDef *hlpf);
float LPF_Calc(LPF_HandleTypeDef *hlpf, float Uk);

/* MSBD function prototypes */
void MSBD_Init_State(MSBD_HandleTypeDef *hmsbd);
float MSBD_Calc(MSBD_HandleTypeDef *hmsbd, float In, uint32_t Time_In);

#endif /* INC_CONTROL_BLOCK_H_ */
