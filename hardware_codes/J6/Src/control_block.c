/**
  ******************************************************************************
  * @file    control_block.c
  * @author	 Kwon Dohyeon
  * @brief   Control block module.
  *          This file provides basic control blocks used in control software,
  *          including:
  *            + PI Controller
  *            + Low-Pass Filter
  *            + Multi-Step Backward Difference
  *            + Other Fundamental Control Functions
  *
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include <stdint.h>
#include "control_block.h"


/* Private typedef -----------------------------------------------------------*/
/*  */


/* Private define ------------------------------------------------------------*/
/*  */


/* Private macros ------------------------------------------------------------*/
/*  */


/* Private variables ---------------------------------------------------------*/
/*  */


/* Private function prototypes -----------------------------------------------*/
/*  */


/* Exported functions --------------------------------------------------------*/
/**
  * @brief  Calculate P controller output with saturation
  * @param  hp  Pointer to a P_HandleTypeDef structure that contains
  *             the configuration of the P controller
  * @param  Ref Reference input
  * @param  Fdb Feedback input
  * @retval P controller output
  */
float P_Calc(P_HandleTypeDef *hp, float Ref, float Fdb)
{
	/* Declaration of local variables */
	float Err, Out;

	/* Error Calculation */
	Err = Ref - Fdb;

	/* Proportional Control */
	Out = Err * hp->Kp;

	/* Implementation Of Saturation */
	Out = MC_SAT(Out, hp->OutMax);

	return Out;
}

/**
  * @brief  Initialize PI controller states
  * @param  hpi Pointer to a PI_HandleTypeDef structure that contains
  *             the configuration and states of the PI controller
  * @retval None
  */
void PI_Init_State(PI_HandleTypeDef *hpi)
{
	hpi->Is_First_Time = 1U;
	hpi->Intg = 0.0f;
	hpi->Time_In_p = 0U;
}

/**
  * @brief  Calculate PI controller output with saturation and anti-windup
  *         by back-calculation
  * @param  hpi     Pointer to a PI_HandleTypeDef structure that contains
  *                 the configuration and states of the PI controller
  * @param  Ref     Reference input
  * @param  Fdb     Feedback input
  * @param  Time_In Current time input in microseconds
  * @retval PI controller output
  */
float PI_Calc(PI_HandleTypeDef *hpi, float Ref, float Fdb, uint32_t Time_In)
{
	/* Preventing Large Time Difference at the first time */
	if(hpi->Is_First_Time != 1U)
	{
		/* declaration of local variables */
		float Err, OutTmp, Out, SatErr;

		/* Error Calculation */
		Err = Ref - Fdb;

		/* Temporary Output Calculation */
		OutTmp = (Err * hpi->Kp) + hpi->Intg;

		/* Implementation Of Saturation */
		Out = MC_SAT(OutTmp, hpi->OutMax);

		/* Integration Including Anti-Windup by Back-Calculation */
		SatErr = OutTmp - Out;
		hpi->Intg += ((Err * hpi->Ki) - (SatErr * hpi->Ka))
				   * (float)(Time_In - hpi->Time_In_p)
				   * 0.000001f;

		/* Update Previous Time Input */
		hpi->Time_In_p = Time_In;

		return Out;
	}
	else
	{
		/* Update Previous Time Input */
		hpi->Time_In_p = Time_In;

		/* Reset First Time Flag */
		hpi->Is_First_Time = 0U;

		return 0.0f;
	}
}

/**
  * @brief  Initialize the state of a first-order LPF
  * @param  hlpf Pointer to an LPF_HandleTypeDef structure that contains
  *              the configuration and states of the LPF
  * @retval None
  */
void LPF_Init_State(LPF_HandleTypeDef *hlpf)
{
	hlpf->Xk = 0.0f;
}

/**
  * @brief  Calculate the gain coefficients of a first-order LPF
  *         using the bilinear transformation
  * @param  hlpf Pointer to an LPF_HandleTypeDef structure that contains
  *              the configuration and states of the LPF
  * @retval None
  */
void LPF_Calc_Gain(LPF_HandleTypeDef *hlpf)
{
  	/* Calculate LPF Gain from Fc, Tsamp */
  	float Wc = 2.0 * PI * hlpf->Fc;
  	hlpf->K1 = (Wc * (float)hlpf->Tsamp * 0.000001) / (Wc * (float)hlpf->Tsamp * 0.000001 + 2.0);
  	hlpf->K2 = (Wc * (float)hlpf->Tsamp * 0.000001 - 2.0) / (Wc * (float)hlpf->Tsamp * 0.000001 + 2.0);
}

/**
  * @brief  Calculate the output of a first-order LPF and update its state
  * @param  hlpf Pointer to an LPF_HandleTypeDef structure that contains
  *              the configuration and states of the LPF
  * @param  Uk   Current input
  * @retval LPF output
  */
float LPF_Calc(LPF_HandleTypeDef *hlpf, float Uk)
{
	/* declaration of local variables */
	float Yk;

  	/* Calculate Output */
	Yk = Uk * hlpf->K1 + hlpf->Xk;

  	/* Update LPF State */
	hlpf->Xk = Uk * hlpf->K1 * (1 - hlpf->K2) + hlpf->Xk * (-hlpf->K2);

	return Yk;
}

/**
  * @brief  Initialize the states of the multi-step backward differentiator
  * @param  hmsbd Pointer to an MSBD_HandleTypeDef structure that contains
  *               the configuration and states of the differentiator
  * @retval None
  */
void MSBD_Init_State(MSBD_HandleTypeDef *hmsbd)
{
	hmsbd->Index = 0U;

	for(uint8_t i = 0; i < 31; i++)
	{
		hmsbd->Buf[hmsbd->Index] = 0.0f;
		hmsbd->Time_Buf[hmsbd->Index] = 0U;
	}
}

/**
  * @brief  Calculate the derivative using a multi-step backward difference
  * @param  hmsbd   Pointer to an MSBD_HandleTypeDef structure that contains
  *                 the configuration and states of the differentiator
  * @param  In      Current input
  * @param  Time_In Current time input
  * @retval Calculated derivative
  */
float MSBD_Calc(MSBD_HandleTypeDef *hmsbd, float In, uint32_t Time_In)
{
	/* declaration of local variables */
	uint8_t Index_p;
	float Out;

    /* Store Current Input & Time Input */
	hmsbd->Buf[hmsbd->Index] = In;
	hmsbd->Time_Buf[hmsbd->Index] = Time_In;

    /* Calculate Index Before Step Number */
	Index_p = (hmsbd->Index - hmsbd->Step_Num + 31) % 31;

    /* Calculate Multi-Step Backward Difference */
	Out = (hmsbd->Buf[hmsbd->Index] - hmsbd->Buf[Index_p])
		/ (float)(hmsbd->Time_Buf[hmsbd->Index] - hmsbd->Time_Buf[Index_p])
		* 1000000.0f;

    /* Update Current Index */
	hmsbd->Index = (hmsbd->Index + 1) % 31;

	return Out;
}


/* Private functions --------------------------------------------------------*/
/**
  * @brief
  * @param
  * @retval
  */
