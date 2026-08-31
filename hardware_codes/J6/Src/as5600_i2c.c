/**
  ******************************************************************************
  * @file    as5600_i2c.c
  * @author  Kwon Dohyeon
  * @brief   I2C for AS5600 initialization module.
  *          This file provides functions to initialize and configure
  *          I2C peripherals.
  *
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "stm32f767xx.h"
#include "register_macro.h"
#include "main.h"
#include "as5600_i2c.h"


/* Private typedef -----------------------------------------------------------*/
/*  */


/* Private define ------------------------------------------------------------*/
/* Address define */
#define AS5600_SADD                (0x36U << 1)
#define AS5600_RAW_ANGLE_ADD        0x0CU

/* Timeout define */
#define AS5600_TIMEOUT_MS           1U

/* SCL Clock Frequency define */
#define AS5600_I2C_TIMING_100KHZ    0x40422F3B
#define AS5600_I2C_TIMING_400KHZ    0x40420B0E


/* Private macros ------------------------------------------------------------*/
/*  */


/* Private variables ---------------------------------------------------------*/
/* I2C Handler variable */
static I2C_HandleTypeDef hi2c2;


/* Private function prototypes -----------------------------------------------*/
/* Recovering function prototype */
static void AS5600_I2C2_Recover(void);


/* Exported functions --------------------------------------------------------*/
/**
  * @brief  Initialize the I2C communication according to the required configuration
  * @param  None
  * @retval None
  */
void AS5600_I2C2_Init(void)
{
	/* 구조체를 가리키는 포인터 자료형의 I2C Base 주소를 구조체 멤버 변수에 저장(레지스터 접근을 위함) */
    hi2c2.Instance = I2C2;

	/*
	 * clock.c 기준:
	 * SYSCLK = 216 MHz
	 * PCLK1  = 54 MHz
	 * I2C2 약 100 kHz 설정
	 */
    hi2c2.Init.Timing = AS5600_I2C_TIMING_400KHZ;

    hi2c2.Init.OwnAddress1 = 0;
    hi2c2.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
    hi2c2.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
    hi2c2.Init.OwnAddress2 = 0;
    hi2c2.Init.OwnAddress2Masks = I2C_OA2_NOMASK;
    hi2c2.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
    hi2c2.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;

    if (HAL_I2C_Init(&hi2c2) != HAL_OK)
    {
        Error_Handler();
    }

    if (HAL_I2CEx_ConfigAnalogFilter(&hi2c2, I2C_ANALOGFILTER_ENABLE) != HAL_OK)
    {
        Error_Handler();
    }

    if (HAL_I2CEx_ConfigDigitalFilter(&hi2c2, 0) != HAL_OK)
    {
        Error_Handler();
    }
}

/**
  * @brief  Read the raw angle data through I2C communication
  * @param  None
  * @retval Raw angle value on success, AS5600_READ_ERROR on failure
  */
uint16_t AS5600_ReadRawAngle(void)
{
    uint8_t rx[2] = {0, 0};
    HAL_StatusTypeDef status;

    status = HAL_I2C_Mem_Read(&hi2c2,
    						  AS5600_SADD,
							  AS5600_RAW_ANGLE_ADD,
                              I2C_MEMADD_SIZE_8BIT,
                              rx,
                              2,
							  AS5600_TIMEOUT_MS);

    if(status == HAL_OK)
    {
    	/* 12 bit resolution verification */
    	if((rx[0] & 0xF0) != 0x00)
    	{
    		return AS5600_READ_ERROR;
    	}
    	else
    	{
    		return (((uint16_t)rx[0] << 8) | (uint16_t)rx[1]);
    	}
    }
    else
    {
    	AS5600_I2C2_Recover();
        return AS5600_READ_ERROR;
    }
}


/* Private functions --------------------------------------------------------*/
/**
  * @brief  Recover the I2C communication state after a communication error
  * @param  None
  * @retval None
  */
static void AS5600_I2C2_Recover(void)
{
    /* I2C2 주변장치 비활성화 */
    I2C2->CR1 &= ~I2C_CR1_PE;

    /* PE = 0 설정이 실제 주변장치에 전달되도록 CR1 읽기 */
    (void)I2C2->CR1;

    /* PE = 0 유지시간 확보 */
    for (volatile uint32_t i = 0U; i < 16U; i++)
    {
        __NOP();
    }

    /* I2C2 주변장치 다시 활성화 */
    I2C2->CR1 |= I2C_CR1_PE;

    /* HAL Handle 상태 복구 */
    hi2c2.State = HAL_I2C_STATE_READY;
    hi2c2.Mode  = HAL_I2C_MODE_NONE;
    hi2c2.Lock  = HAL_UNLOCKED;
}
