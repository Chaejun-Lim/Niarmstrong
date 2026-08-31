/**
  ******************************************************************************
  * @file    gpio.c
  * @author  Kwon Dohyeon
  * @brief   MPU initialization module.
  *          This file provides functions to initialize and configure
  *          MPU peripherals.
  *
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "system_config.h"


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
  * @brief  Initialize MPU
  * @param  None
  * @retval None
  */
void MPU_Config(void)
{
    MPU_Region_InitTypeDef MPU_InitStruct = {0};

    HAL_MPU_Disable();

    MPU_InitStruct.Enable = MPU_REGION_ENABLE;
    MPU_InitStruct.Number = MPU_REGION_NUMBER0;
    MPU_InitStruct.BaseAddress = 0x0;
    MPU_InitStruct.Size = MPU_REGION_SIZE_4GB;
    MPU_InitStruct.SubRegionDisable = 0x87;
    MPU_InitStruct.TypeExtField = MPU_TEX_LEVEL0;
    MPU_InitStruct.AccessPermission = MPU_REGION_NO_ACCESS;
    MPU_InitStruct.DisableExec = MPU_INSTRUCTION_ACCESS_DISABLE;
    MPU_InitStruct.IsShareable = MPU_ACCESS_SHAREABLE;
    MPU_InitStruct.IsCacheable = MPU_ACCESS_NOT_CACHEABLE;
    MPU_InitStruct.IsBufferable = MPU_ACCESS_NOT_BUFFERABLE;

    HAL_MPU_ConfigRegion(&MPU_InitStruct);
    HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);
}


/* Private functions --------------------------------------------------------*/
/**
  * @brief
  * @param
  * @retval
  */
