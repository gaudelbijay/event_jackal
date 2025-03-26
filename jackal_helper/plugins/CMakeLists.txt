cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

find_package(gazebo REQUIRED)
include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")


add_library(ActorCollisionsPlugin SHARED ActorCollisionsPlugin.cc)
target_link_libraries(ActorCollisionsPlugin ${GAZEBO_LIBRARIES})

add_library(R2L_74_0 SHARED R2L_74_0.cc)
target_link_libraries(R2L_74_0 ${GAZEBO_LIBRARIES})

add_library(R2L_77_1 SHARED R2L_77_1.cc)
target_link_libraries(R2L_77_1 ${GAZEBO_LIBRARIES})

add_library(R2L_141_2 SHARED R2L_141_2.cc)
target_link_libraries(R2L_141_2 ${GAZEBO_LIBRARIES})

add_library(R2L_25_3 SHARED R2L_25_3.cc)
target_link_libraries(R2L_25_3 ${GAZEBO_LIBRARIES})

add_library(R2L_33_4 SHARED R2L_33_4.cc)
target_link_libraries(R2L_33_4 ${GAZEBO_LIBRARIES})

add_library(R2L_26_5 SHARED R2L_26_5.cc)
target_link_libraries(R2L_26_5 ${GAZEBO_LIBRARIES})

add_library(R2L_116_6 SHARED R2L_116_6.cc)
target_link_libraries(R2L_116_6 ${GAZEBO_LIBRARIES})

add_library(R2L_54_7 SHARED R2L_54_7.cc)
target_link_libraries(R2L_54_7 ${GAZEBO_LIBRARIES})

add_library(R2L_109_8 SHARED R2L_109_8.cc)
target_link_libraries(R2L_109_8 ${GAZEBO_LIBRARIES})

add_library(R2L_77_9 SHARED R2L_77_9.cc)
target_link_libraries(R2L_77_9 ${GAZEBO_LIBRARIES})

add_library(R2L_110_10 SHARED R2L_110_10.cc)
target_link_libraries(R2L_110_10 ${GAZEBO_LIBRARIES})

add_library(R2L_142_11 SHARED R2L_142_11.cc)
target_link_libraries(R2L_142_11 ${GAZEBO_LIBRARIES})

add_library(R2L_43_12 SHARED R2L_43_12.cc)
target_link_libraries(R2L_43_12 ${GAZEBO_LIBRARIES})

add_library(R2L_61_13 SHARED R2L_61_13.cc)
target_link_libraries(R2L_61_13 ${GAZEBO_LIBRARIES})

add_library(R2L_19_14 SHARED R2L_19_14.cc)
target_link_libraries(R2L_19_14 ${GAZEBO_LIBRARIES})

add_library(R2L_136_15 SHARED R2L_136_15.cc)
target_link_libraries(R2L_136_15 ${GAZEBO_LIBRARIES})

add_library(R2L_96_16 SHARED R2L_96_16.cc)
target_link_libraries(R2L_96_16 ${GAZEBO_LIBRARIES})

add_library(R2L_71_17 SHARED R2L_71_17.cc)
target_link_libraries(R2L_71_17 ${GAZEBO_LIBRARIES})

add_library(R2L_12_18 SHARED R2L_12_18.cc)
target_link_libraries(R2L_12_18 ${GAZEBO_LIBRARIES})

add_library(R2L_108_19 SHARED R2L_108_19.cc)
target_link_libraries(R2L_108_19 ${GAZEBO_LIBRARIES})

add_library(L2R_66_0 SHARED L2R_66_0.cc)
target_link_libraries(L2R_66_0 ${GAZEBO_LIBRARIES})

add_library(L2R_117_1 SHARED L2R_117_1.cc)
target_link_libraries(L2R_117_1 ${GAZEBO_LIBRARIES})

add_library(L2R_99_2 SHARED L2R_99_2.cc)
target_link_libraries(L2R_99_2 ${GAZEBO_LIBRARIES})

add_library(L2R_113_3 SHARED L2R_113_3.cc)
target_link_libraries(L2R_113_3 ${GAZEBO_LIBRARIES})

add_library(L2R_95_4 SHARED L2R_95_4.cc)
target_link_libraries(L2R_95_4 ${GAZEBO_LIBRARIES})

add_library(L2R_20_5 SHARED L2R_20_5.cc)
target_link_libraries(L2R_20_5 ${GAZEBO_LIBRARIES})

add_library(L2R_73_6 SHARED L2R_73_6.cc)
target_link_libraries(L2R_73_6 ${GAZEBO_LIBRARIES})

add_library(L2R_68_7 SHARED L2R_68_7.cc)
target_link_libraries(L2R_68_7 ${GAZEBO_LIBRARIES})

add_library(L2R_121_8 SHARED L2R_121_8.cc)
target_link_libraries(L2R_121_8 ${GAZEBO_LIBRARIES})

add_library(L2R_132_9 SHARED L2R_132_9.cc)
target_link_libraries(L2R_132_9 ${GAZEBO_LIBRARIES})

add_library(L2R_37_10 SHARED L2R_37_10.cc)
target_link_libraries(L2R_37_10 ${GAZEBO_LIBRARIES})

add_library(L2R_71_11 SHARED L2R_71_11.cc)
target_link_libraries(L2R_71_11 ${GAZEBO_LIBRARIES})

add_library(L2R_102_12 SHARED L2R_102_12.cc)
target_link_libraries(L2R_102_12 ${GAZEBO_LIBRARIES})

add_library(L2R_40_13 SHARED L2R_40_13.cc)
target_link_libraries(L2R_40_13 ${GAZEBO_LIBRARIES})

add_library(L2R_71_14 SHARED L2R_71_14.cc)
target_link_libraries(L2R_71_14 ${GAZEBO_LIBRARIES})

add_library(L2R_21_15 SHARED L2R_21_15.cc)
target_link_libraries(L2R_21_15 ${GAZEBO_LIBRARIES})

add_library(L2R_146_16 SHARED L2R_146_16.cc)
target_link_libraries(L2R_146_16 ${GAZEBO_LIBRARIES})

add_library(L2R_101_17 SHARED L2R_101_17.cc)
target_link_libraries(L2R_101_17 ${GAZEBO_LIBRARIES})

add_library(L2R_84_18 SHARED L2R_84_18.cc)
target_link_libraries(L2R_84_18 ${GAZEBO_LIBRARIES})

add_library(L2R_50_19 SHARED L2R_50_19.cc)
target_link_libraries(L2R_50_19 ${GAZEBO_LIBRARIES})

add_library(T2B_111_0 SHARED T2B_111_0.cc)
target_link_libraries(T2B_111_0 ${GAZEBO_LIBRARIES})

add_library(T2B_100_1 SHARED T2B_100_1.cc)
target_link_libraries(T2B_100_1 ${GAZEBO_LIBRARIES})

add_library(T2B_145_2 SHARED T2B_145_2.cc)
target_link_libraries(T2B_145_2 ${GAZEBO_LIBRARIES})

add_library(T2B_27_3 SHARED T2B_27_3.cc)
target_link_libraries(T2B_27_3 ${GAZEBO_LIBRARIES})

add_library(T2B_137_4 SHARED T2B_137_4.cc)
target_link_libraries(T2B_137_4 ${GAZEBO_LIBRARIES})

add_library(T2B_43_5 SHARED T2B_43_5.cc)
target_link_libraries(T2B_43_5 ${GAZEBO_LIBRARIES})

add_library(T2B_93_6 SHARED T2B_93_6.cc)
target_link_libraries(T2B_93_6 ${GAZEBO_LIBRARIES})

add_library(T2B_41_7 SHARED T2B_41_7.cc)
target_link_libraries(T2B_41_7 ${GAZEBO_LIBRARIES})

add_library(T2B_80_8 SHARED T2B_80_8.cc)
target_link_libraries(T2B_80_8 ${GAZEBO_LIBRARIES})

add_library(T2B_140_9 SHARED T2B_140_9.cc)
target_link_libraries(T2B_140_9 ${GAZEBO_LIBRARIES})

add_library(T2B_60_10 SHARED T2B_60_10.cc)
target_link_libraries(T2B_60_10 ${GAZEBO_LIBRARIES})

add_library(T2B_113_11 SHARED T2B_113_11.cc)
target_link_libraries(T2B_113_11 ${GAZEBO_LIBRARIES})

add_library(T2B_42_12 SHARED T2B_42_12.cc)
target_link_libraries(T2B_42_12 ${GAZEBO_LIBRARIES})

add_library(T2B_128_13 SHARED T2B_128_13.cc)
target_link_libraries(T2B_128_13 ${GAZEBO_LIBRARIES})

add_library(T2B_17_14 SHARED T2B_17_14.cc)
target_link_libraries(T2B_17_14 ${GAZEBO_LIBRARIES})

add_library(T2B_73_15 SHARED T2B_73_15.cc)
target_link_libraries(T2B_73_15 ${GAZEBO_LIBRARIES})

add_library(T2B_59_16 SHARED T2B_59_16.cc)
target_link_libraries(T2B_59_16 ${GAZEBO_LIBRARIES})

add_library(T2B_53_17 SHARED T2B_53_17.cc)
target_link_libraries(T2B_53_17 ${GAZEBO_LIBRARIES})

add_library(T2B_60_18 SHARED T2B_60_18.cc)
target_link_libraries(T2B_60_18 ${GAZEBO_LIBRARIES})

add_library(T2B_126_19 SHARED T2B_126_19.cc)
target_link_libraries(T2B_126_19 ${GAZEBO_LIBRARIES})

    
add_library(thin_R2L_46_0 SHARED thin_R2L_46_0.cc)
target_link_libraries(thin_R2L_46_0 ${GAZEBO_LIBRARIES})

add_library(thin_R2L_138_1 SHARED thin_R2L_138_1.cc)
target_link_libraries(thin_R2L_138_1 ${GAZEBO_LIBRARIES})

add_library(thin_R2L_143_2 SHARED thin_R2L_143_2.cc)
target_link_libraries(thin_R2L_143_2 ${GAZEBO_LIBRARIES})

add_library(thin_R2L_94_3 SHARED thin_R2L_94_3.cc)
target_link_libraries(thin_R2L_94_3 ${GAZEBO_LIBRARIES})

add_library(thin_R2L_10_4 SHARED thin_R2L_10_4.cc)
target_link_libraries(thin_R2L_10_4 ${GAZEBO_LIBRARIES})

add_library(thin_R2L_77_5 SHARED thin_R2L_77_5.cc)
target_link_libraries(thin_R2L_77_5 ${GAZEBO_LIBRARIES})

add_library(thin_R2L_117_6 SHARED thin_R2L_117_6.cc)
target_link_libraries(thin_R2L_117_6 ${GAZEBO_LIBRARIES})

add_library(thin_R2L_26_7 SHARED thin_R2L_26_7.cc)
target_link_libraries(thin_R2L_26_7 ${GAZEBO_LIBRARIES})

add_library(thin_R2L_75_8 SHARED thin_R2L_75_8.cc)
target_link_libraries(thin_R2L_75_8 ${GAZEBO_LIBRARIES})

add_library(thin_R2L_112_9 SHARED thin_R2L_112_9.cc)
target_link_libraries(thin_R2L_112_9 ${GAZEBO_LIBRARIES})

add_library(thin_R2L_56_10 SHARED thin_R2L_56_10.cc)
target_link_libraries(thin_R2L_56_10 ${GAZEBO_LIBRARIES})

add_library(thin_R2L_143_11 SHARED thin_R2L_143_11.cc)
target_link_libraries(thin_R2L_143_11 ${GAZEBO_LIBRARIES})

add_library(thin_R2L_66_12 SHARED thin_R2L_66_12.cc)
target_link_libraries(thin_R2L_66_12 ${GAZEBO_LIBRARIES})

add_library(thin_R2L_149_13 SHARED thin_R2L_149_13.cc)
target_link_libraries(thin_R2L_149_13 ${GAZEBO_LIBRARIES})

add_library(thin_R2L_68_14 SHARED thin_R2L_68_14.cc)
target_link_libraries(thin_R2L_68_14 ${GAZEBO_LIBRARIES})

add_library(thin_R2L_62_15 SHARED thin_R2L_62_15.cc)
target_link_libraries(thin_R2L_62_15 ${GAZEBO_LIBRARIES})

add_library(thin_R2L_21_16 SHARED thin_R2L_21_16.cc)
target_link_libraries(thin_R2L_21_16 ${GAZEBO_LIBRARIES})

add_library(thin_R2L_13_17 SHARED thin_R2L_13_17.cc)
target_link_libraries(thin_R2L_13_17 ${GAZEBO_LIBRARIES})

add_library(thin_R2L_17_18 SHARED thin_R2L_17_18.cc)
target_link_libraries(thin_R2L_17_18 ${GAZEBO_LIBRARIES})

add_library(thin_R2L_16_19 SHARED thin_R2L_16_19.cc)
target_link_libraries(thin_R2L_16_19 ${GAZEBO_LIBRARIES})

add_library(thin_L2R_25_0 SHARED thin_L2R_25_0.cc)
target_link_libraries(thin_L2R_25_0 ${GAZEBO_LIBRARIES})

add_library(thin_L2R_78_1 SHARED thin_L2R_78_1.cc)
target_link_libraries(thin_L2R_78_1 ${GAZEBO_LIBRARIES})

add_library(thin_L2R_135_2 SHARED thin_L2R_135_2.cc)
target_link_libraries(thin_L2R_135_2 ${GAZEBO_LIBRARIES})

add_library(thin_L2R_72_3 SHARED thin_L2R_72_3.cc)
target_link_libraries(thin_L2R_72_3 ${GAZEBO_LIBRARIES})

add_library(thin_L2R_39_4 SHARED thin_L2R_39_4.cc)
target_link_libraries(thin_L2R_39_4 ${GAZEBO_LIBRARIES})

add_library(thin_L2R_93_5 SHARED thin_L2R_93_5.cc)
target_link_libraries(thin_L2R_93_5 ${GAZEBO_LIBRARIES})

add_library(thin_L2R_99_6 SHARED thin_L2R_99_6.cc)
target_link_libraries(thin_L2R_99_6 ${GAZEBO_LIBRARIES})

add_library(thin_L2R_72_7 SHARED thin_L2R_72_7.cc)
target_link_libraries(thin_L2R_72_7 ${GAZEBO_LIBRARIES})

add_library(thin_L2R_96_8 SHARED thin_L2R_96_8.cc)
target_link_libraries(thin_L2R_96_8 ${GAZEBO_LIBRARIES})

add_library(thin_L2R_50_9 SHARED thin_L2R_50_9.cc)
target_link_libraries(thin_L2R_50_9 ${GAZEBO_LIBRARIES})

add_library(thin_L2R_142_10 SHARED thin_L2R_142_10.cc)
target_link_libraries(thin_L2R_142_10 ${GAZEBO_LIBRARIES})

add_library(thin_L2R_10_11 SHARED thin_L2R_10_11.cc)
target_link_libraries(thin_L2R_10_11 ${GAZEBO_LIBRARIES})

add_library(thin_L2R_10_12 SHARED thin_L2R_10_12.cc)
target_link_libraries(thin_L2R_10_12 ${GAZEBO_LIBRARIES})

add_library(thin_L2R_93_13 SHARED thin_L2R_93_13.cc)
target_link_libraries(thin_L2R_93_13 ${GAZEBO_LIBRARIES})

add_library(thin_L2R_53_14 SHARED thin_L2R_53_14.cc)
target_link_libraries(thin_L2R_53_14 ${GAZEBO_LIBRARIES})

add_library(thin_L2R_137_15 SHARED thin_L2R_137_15.cc)
target_link_libraries(thin_L2R_137_15 ${GAZEBO_LIBRARIES})

add_library(thin_L2R_62_16 SHARED thin_L2R_62_16.cc)
target_link_libraries(thin_L2R_62_16 ${GAZEBO_LIBRARIES})

add_library(thin_L2R_38_17 SHARED thin_L2R_38_17.cc)
target_link_libraries(thin_L2R_38_17 ${GAZEBO_LIBRARIES})

add_library(thin_L2R_81_18 SHARED thin_L2R_81_18.cc)
target_link_libraries(thin_L2R_81_18 ${GAZEBO_LIBRARIES})

add_library(thin_L2R_28_19 SHARED thin_L2R_28_19.cc)
target_link_libraries(thin_L2R_28_19 ${GAZEBO_LIBRARIES})

add_library(thin_T2B_120_0 SHARED thin_T2B_120_0.cc)
target_link_libraries(thin_T2B_120_0 ${GAZEBO_LIBRARIES})

add_library(thin_T2B_140_1 SHARED thin_T2B_140_1.cc)
target_link_libraries(thin_T2B_140_1 ${GAZEBO_LIBRARIES})

add_library(thin_T2B_50_2 SHARED thin_T2B_50_2.cc)
target_link_libraries(thin_T2B_50_2 ${GAZEBO_LIBRARIES})

add_library(thin_T2B_19_3 SHARED thin_T2B_19_3.cc)
target_link_libraries(thin_T2B_19_3 ${GAZEBO_LIBRARIES})

add_library(thin_T2B_34_4 SHARED thin_T2B_34_4.cc)
target_link_libraries(thin_T2B_34_4 ${GAZEBO_LIBRARIES})

add_library(thin_T2B_93_5 SHARED thin_T2B_93_5.cc)
target_link_libraries(thin_T2B_93_5 ${GAZEBO_LIBRARIES})

add_library(thin_T2B_145_6 SHARED thin_T2B_145_6.cc)
target_link_libraries(thin_T2B_145_6 ${GAZEBO_LIBRARIES})

add_library(thin_T2B_33_7 SHARED thin_T2B_33_7.cc)
target_link_libraries(thin_T2B_33_7 ${GAZEBO_LIBRARIES})

add_library(thin_T2B_78_8 SHARED thin_T2B_78_8.cc)
target_link_libraries(thin_T2B_78_8 ${GAZEBO_LIBRARIES})

add_library(thin_T2B_113_9 SHARED thin_T2B_113_9.cc)
target_link_libraries(thin_T2B_113_9 ${GAZEBO_LIBRARIES})

add_library(thin_T2B_30_10 SHARED thin_T2B_30_10.cc)
target_link_libraries(thin_T2B_30_10 ${GAZEBO_LIBRARIES})

add_library(thin_T2B_51_11 SHARED thin_T2B_51_11.cc)
target_link_libraries(thin_T2B_51_11 ${GAZEBO_LIBRARIES})

add_library(thin_T2B_58_12 SHARED thin_T2B_58_12.cc)
target_link_libraries(thin_T2B_58_12 ${GAZEBO_LIBRARIES})

add_library(thin_T2B_96_13 SHARED thin_T2B_96_13.cc)
target_link_libraries(thin_T2B_96_13 ${GAZEBO_LIBRARIES})

add_library(thin_T2B_118_14 SHARED thin_T2B_118_14.cc)
target_link_libraries(thin_T2B_118_14 ${GAZEBO_LIBRARIES})

add_library(thin_T2B_26_15 SHARED thin_T2B_26_15.cc)
target_link_libraries(thin_T2B_26_15 ${GAZEBO_LIBRARIES})

add_library(thin_T2B_98_16 SHARED thin_T2B_98_16.cc)
target_link_libraries(thin_T2B_98_16 ${GAZEBO_LIBRARIES})

add_library(thin_T2B_55_17 SHARED thin_T2B_55_17.cc)
target_link_libraries(thin_T2B_55_17 ${GAZEBO_LIBRARIES})

add_library(thin_T2B_114_18 SHARED thin_T2B_114_18.cc)
target_link_libraries(thin_T2B_114_18 ${GAZEBO_LIBRARIES})

add_library(thin_T2B_142_19 SHARED thin_T2B_142_19.cc)
target_link_libraries(thin_T2B_142_19 ${GAZEBO_LIBRARIES})

    
add_library(wide_R2L_125_0 SHARED wide_R2L_125_0.cc)
target_link_libraries(wide_R2L_125_0 ${GAZEBO_LIBRARIES})

add_library(wide_R2L_73_1 SHARED wide_R2L_73_1.cc)
target_link_libraries(wide_R2L_73_1 ${GAZEBO_LIBRARIES})

add_library(wide_R2L_99_2 SHARED wide_R2L_99_2.cc)
target_link_libraries(wide_R2L_99_2 ${GAZEBO_LIBRARIES})

add_library(wide_R2L_51_3 SHARED wide_R2L_51_3.cc)
target_link_libraries(wide_R2L_51_3 ${GAZEBO_LIBRARIES})

add_library(wide_R2L_62_4 SHARED wide_R2L_62_4.cc)
target_link_libraries(wide_R2L_62_4 ${GAZEBO_LIBRARIES})

add_library(wide_R2L_58_5 SHARED wide_R2L_58_5.cc)
target_link_libraries(wide_R2L_58_5 ${GAZEBO_LIBRARIES})

add_library(wide_R2L_142_6 SHARED wide_R2L_142_6.cc)
target_link_libraries(wide_R2L_142_6 ${GAZEBO_LIBRARIES})

add_library(wide_R2L_138_7 SHARED wide_R2L_138_7.cc)
target_link_libraries(wide_R2L_138_7 ${GAZEBO_LIBRARIES})

add_library(wide_R2L_98_8 SHARED wide_R2L_98_8.cc)
target_link_libraries(wide_R2L_98_8 ${GAZEBO_LIBRARIES})

add_library(wide_R2L_114_9 SHARED wide_R2L_114_9.cc)
target_link_libraries(wide_R2L_114_9 ${GAZEBO_LIBRARIES})

add_library(wide_R2L_101_10 SHARED wide_R2L_101_10.cc)
target_link_libraries(wide_R2L_101_10 ${GAZEBO_LIBRARIES})

add_library(wide_R2L_143_11 SHARED wide_R2L_143_11.cc)
target_link_libraries(wide_R2L_143_11 ${GAZEBO_LIBRARIES})

add_library(wide_R2L_109_12 SHARED wide_R2L_109_12.cc)
target_link_libraries(wide_R2L_109_12 ${GAZEBO_LIBRARIES})

add_library(wide_R2L_107_13 SHARED wide_R2L_107_13.cc)
target_link_libraries(wide_R2L_107_13 ${GAZEBO_LIBRARIES})

add_library(wide_R2L_74_14 SHARED wide_R2L_74_14.cc)
target_link_libraries(wide_R2L_74_14 ${GAZEBO_LIBRARIES})

add_library(wide_R2L_94_15 SHARED wide_R2L_94_15.cc)
target_link_libraries(wide_R2L_94_15 ${GAZEBO_LIBRARIES})

add_library(wide_R2L_106_16 SHARED wide_R2L_106_16.cc)
target_link_libraries(wide_R2L_106_16 ${GAZEBO_LIBRARIES})

add_library(wide_R2L_23_17 SHARED wide_R2L_23_17.cc)
target_link_libraries(wide_R2L_23_17 ${GAZEBO_LIBRARIES})

add_library(wide_R2L_139_18 SHARED wide_R2L_139_18.cc)
target_link_libraries(wide_R2L_139_18 ${GAZEBO_LIBRARIES})

add_library(wide_R2L_89_19 SHARED wide_R2L_89_19.cc)
target_link_libraries(wide_R2L_89_19 ${GAZEBO_LIBRARIES})

add_library(wide_L2R_127_0 SHARED wide_L2R_127_0.cc)
target_link_libraries(wide_L2R_127_0 ${GAZEBO_LIBRARIES})

add_library(wide_L2R_36_1 SHARED wide_L2R_36_1.cc)
target_link_libraries(wide_L2R_36_1 ${GAZEBO_LIBRARIES})

add_library(wide_L2R_147_2 SHARED wide_L2R_147_2.cc)
target_link_libraries(wide_L2R_147_2 ${GAZEBO_LIBRARIES})

add_library(wide_L2R_24_3 SHARED wide_L2R_24_3.cc)
target_link_libraries(wide_L2R_24_3 ${GAZEBO_LIBRARIES})

add_library(wide_L2R_41_4 SHARED wide_L2R_41_4.cc)
target_link_libraries(wide_L2R_41_4 ${GAZEBO_LIBRARIES})

add_library(wide_L2R_43_5 SHARED wide_L2R_43_5.cc)
target_link_libraries(wide_L2R_43_5 ${GAZEBO_LIBRARIES})

add_library(wide_L2R_144_6 SHARED wide_L2R_144_6.cc)
target_link_libraries(wide_L2R_144_6 ${GAZEBO_LIBRARIES})

add_library(wide_L2R_116_7 SHARED wide_L2R_116_7.cc)
target_link_libraries(wide_L2R_116_7 ${GAZEBO_LIBRARIES})

add_library(wide_L2R_71_8 SHARED wide_L2R_71_8.cc)
target_link_libraries(wide_L2R_71_8 ${GAZEBO_LIBRARIES})

add_library(wide_L2R_30_9 SHARED wide_L2R_30_9.cc)
target_link_libraries(wide_L2R_30_9 ${GAZEBO_LIBRARIES})

add_library(wide_L2R_25_10 SHARED wide_L2R_25_10.cc)
target_link_libraries(wide_L2R_25_10 ${GAZEBO_LIBRARIES})

add_library(wide_L2R_37_11 SHARED wide_L2R_37_11.cc)
target_link_libraries(wide_L2R_37_11 ${GAZEBO_LIBRARIES})

add_library(wide_L2R_101_12 SHARED wide_L2R_101_12.cc)
target_link_libraries(wide_L2R_101_12 ${GAZEBO_LIBRARIES})

add_library(wide_L2R_144_13 SHARED wide_L2R_144_13.cc)
target_link_libraries(wide_L2R_144_13 ${GAZEBO_LIBRARIES})

add_library(wide_L2R_128_14 SHARED wide_L2R_128_14.cc)
target_link_libraries(wide_L2R_128_14 ${GAZEBO_LIBRARIES})

add_library(wide_L2R_31_15 SHARED wide_L2R_31_15.cc)
target_link_libraries(wide_L2R_31_15 ${GAZEBO_LIBRARIES})

add_library(wide_L2R_114_16 SHARED wide_L2R_114_16.cc)
target_link_libraries(wide_L2R_114_16 ${GAZEBO_LIBRARIES})

add_library(wide_L2R_41_17 SHARED wide_L2R_41_17.cc)
target_link_libraries(wide_L2R_41_17 ${GAZEBO_LIBRARIES})

add_library(wide_L2R_45_18 SHARED wide_L2R_45_18.cc)
target_link_libraries(wide_L2R_45_18 ${GAZEBO_LIBRARIES})

add_library(wide_L2R_41_19 SHARED wide_L2R_41_19.cc)
target_link_libraries(wide_L2R_41_19 ${GAZEBO_LIBRARIES})

add_library(wide_T2B_85_0 SHARED wide_T2B_85_0.cc)
target_link_libraries(wide_T2B_85_0 ${GAZEBO_LIBRARIES})

add_library(wide_T2B_126_1 SHARED wide_T2B_126_1.cc)
target_link_libraries(wide_T2B_126_1 ${GAZEBO_LIBRARIES})

add_library(wide_T2B_34_2 SHARED wide_T2B_34_2.cc)
target_link_libraries(wide_T2B_34_2 ${GAZEBO_LIBRARIES})

add_library(wide_T2B_58_3 SHARED wide_T2B_58_3.cc)
target_link_libraries(wide_T2B_58_3 ${GAZEBO_LIBRARIES})

add_library(wide_T2B_88_4 SHARED wide_T2B_88_4.cc)
target_link_libraries(wide_T2B_88_4 ${GAZEBO_LIBRARIES})

add_library(wide_T2B_114_5 SHARED wide_T2B_114_5.cc)
target_link_libraries(wide_T2B_114_5 ${GAZEBO_LIBRARIES})

add_library(wide_T2B_19_6 SHARED wide_T2B_19_6.cc)
target_link_libraries(wide_T2B_19_6 ${GAZEBO_LIBRARIES})

add_library(wide_T2B_15_7 SHARED wide_T2B_15_7.cc)
target_link_libraries(wide_T2B_15_7 ${GAZEBO_LIBRARIES})

add_library(wide_T2B_48_8 SHARED wide_T2B_48_8.cc)
target_link_libraries(wide_T2B_48_8 ${GAZEBO_LIBRARIES})

add_library(wide_T2B_29_9 SHARED wide_T2B_29_9.cc)
target_link_libraries(wide_T2B_29_9 ${GAZEBO_LIBRARIES})

add_library(wide_T2B_91_10 SHARED wide_T2B_91_10.cc)
target_link_libraries(wide_T2B_91_10 ${GAZEBO_LIBRARIES})

add_library(wide_T2B_49_11 SHARED wide_T2B_49_11.cc)
target_link_libraries(wide_T2B_49_11 ${GAZEBO_LIBRARIES})

add_library(wide_T2B_27_12 SHARED wide_T2B_27_12.cc)
target_link_libraries(wide_T2B_27_12 ${GAZEBO_LIBRARIES})

add_library(wide_T2B_64_13 SHARED wide_T2B_64_13.cc)
target_link_libraries(wide_T2B_64_13 ${GAZEBO_LIBRARIES})

add_library(wide_T2B_137_14 SHARED wide_T2B_137_14.cc)
target_link_libraries(wide_T2B_137_14 ${GAZEBO_LIBRARIES})

add_library(wide_T2B_23_15 SHARED wide_T2B_23_15.cc)
target_link_libraries(wide_T2B_23_15 ${GAZEBO_LIBRARIES})

add_library(wide_T2B_136_16 SHARED wide_T2B_136_16.cc)
target_link_libraries(wide_T2B_136_16 ${GAZEBO_LIBRARIES})

add_library(wide_T2B_149_17 SHARED wide_T2B_149_17.cc)
target_link_libraries(wide_T2B_149_17 ${GAZEBO_LIBRARIES})

add_library(wide_T2B_56_18 SHARED wide_T2B_56_18.cc)
target_link_libraries(wide_T2B_56_18 ${GAZEBO_LIBRARIES})

add_library(wide_T2B_103_19 SHARED wide_T2B_103_19.cc)
target_link_libraries(wide_T2B_103_19 ${GAZEBO_LIBRARIES})


add_library(wall_R2L_122_0 SHARED wall_R2L_122_0.cc)
target_link_libraries(wall_R2L_122_0 ${GAZEBO_LIBRARIES})

add_library(wall_R2L_155_1 SHARED wall_R2L_155_1.cc)
target_link_libraries(wall_R2L_155_1 ${GAZEBO_LIBRARIES})

add_library(wall_R2L_136_2 SHARED wall_R2L_136_2.cc)
target_link_libraries(wall_R2L_136_2 ${GAZEBO_LIBRARIES})

add_library(wall_R2L_168_3 SHARED wall_R2L_168_3.cc)
target_link_libraries(wall_R2L_168_3 ${GAZEBO_LIBRARIES})

add_library(wall_R2L_107_4 SHARED wall_R2L_107_4.cc)
target_link_libraries(wall_R2L_107_4 ${GAZEBO_LIBRARIES})

add_library(wall_R2L_156_5 SHARED wall_R2L_156_5.cc)
target_link_libraries(wall_R2L_156_5 ${GAZEBO_LIBRARIES})

add_library(wall_R2L_172_6 SHARED wall_R2L_172_6.cc)
target_link_libraries(wall_R2L_172_6 ${GAZEBO_LIBRARIES})

add_library(wall_R2L_125_7 SHARED wall_R2L_125_7.cc)
target_link_libraries(wall_R2L_125_7 ${GAZEBO_LIBRARIES})

add_library(wall_R2L_169_8 SHARED wall_R2L_169_8.cc)
target_link_libraries(wall_R2L_169_8 ${GAZEBO_LIBRARIES})

add_library(wall_R2L_121_9 SHARED wall_R2L_121_9.cc)
target_link_libraries(wall_R2L_121_9 ${GAZEBO_LIBRARIES})

add_library(wall_R2L_112_10 SHARED wall_R2L_112_10.cc)
target_link_libraries(wall_R2L_112_10 ${GAZEBO_LIBRARIES})

add_library(wall_R2L_156_11 SHARED wall_R2L_156_11.cc)
target_link_libraries(wall_R2L_156_11 ${GAZEBO_LIBRARIES})

add_library(wall_R2L_131_12 SHARED wall_R2L_131_12.cc)
target_link_libraries(wall_R2L_131_12 ${GAZEBO_LIBRARIES})

add_library(wall_R2L_175_13 SHARED wall_R2L_175_13.cc)
target_link_libraries(wall_R2L_175_13 ${GAZEBO_LIBRARIES})

add_library(wall_R2L_176_14 SHARED wall_R2L_176_14.cc)
target_link_libraries(wall_R2L_176_14 ${GAZEBO_LIBRARIES})

add_library(wall_R2L_107_15 SHARED wall_R2L_107_15.cc)
target_link_libraries(wall_R2L_107_15 ${GAZEBO_LIBRARIES})

add_library(wall_R2L_105_16 SHARED wall_R2L_105_16.cc)
target_link_libraries(wall_R2L_105_16 ${GAZEBO_LIBRARIES})

add_library(wall_R2L_144_17 SHARED wall_R2L_144_17.cc)
target_link_libraries(wall_R2L_144_17 ${GAZEBO_LIBRARIES})

add_library(wall_R2L_166_18 SHARED wall_R2L_166_18.cc)
target_link_libraries(wall_R2L_166_18 ${GAZEBO_LIBRARIES})

add_library(wall_R2L_126_19 SHARED wall_R2L_126_19.cc)
target_link_libraries(wall_R2L_126_19 ${GAZEBO_LIBRARIES})

add_library(wall_L2R_177_0 SHARED wall_L2R_177_0.cc)
target_link_libraries(wall_L2R_177_0 ${GAZEBO_LIBRARIES})

add_library(wall_L2R_113_1 SHARED wall_L2R_113_1.cc)
target_link_libraries(wall_L2R_113_1 ${GAZEBO_LIBRARIES})

add_library(wall_L2R_134_2 SHARED wall_L2R_134_2.cc)
target_link_libraries(wall_L2R_134_2 ${GAZEBO_LIBRARIES})

add_library(wall_L2R_119_3 SHARED wall_L2R_119_3.cc)
target_link_libraries(wall_L2R_119_3 ${GAZEBO_LIBRARIES})

add_library(wall_L2R_197_4 SHARED wall_L2R_197_4.cc)
target_link_libraries(wall_L2R_197_4 ${GAZEBO_LIBRARIES})

add_library(wall_L2R_135_5 SHARED wall_L2R_135_5.cc)
target_link_libraries(wall_L2R_135_5 ${GAZEBO_LIBRARIES})

add_library(wall_L2R_161_6 SHARED wall_L2R_161_6.cc)
target_link_libraries(wall_L2R_161_6 ${GAZEBO_LIBRARIES})

add_library(wall_L2R_174_7 SHARED wall_L2R_174_7.cc)
target_link_libraries(wall_L2R_174_7 ${GAZEBO_LIBRARIES})

add_library(wall_L2R_142_8 SHARED wall_L2R_142_8.cc)
target_link_libraries(wall_L2R_142_8 ${GAZEBO_LIBRARIES})

add_library(wall_L2R_197_9 SHARED wall_L2R_197_9.cc)
target_link_libraries(wall_L2R_197_9 ${GAZEBO_LIBRARIES})

add_library(wall_L2R_143_10 SHARED wall_L2R_143_10.cc)
target_link_libraries(wall_L2R_143_10 ${GAZEBO_LIBRARIES})

add_library(wall_L2R_104_11 SHARED wall_L2R_104_11.cc)
target_link_libraries(wall_L2R_104_11 ${GAZEBO_LIBRARIES})

add_library(wall_L2R_167_12 SHARED wall_L2R_167_12.cc)
target_link_libraries(wall_L2R_167_12 ${GAZEBO_LIBRARIES})

add_library(wall_L2R_156_13 SHARED wall_L2R_156_13.cc)
target_link_libraries(wall_L2R_156_13 ${GAZEBO_LIBRARIES})

add_library(wall_L2R_168_14 SHARED wall_L2R_168_14.cc)
target_link_libraries(wall_L2R_168_14 ${GAZEBO_LIBRARIES})

add_library(wall_L2R_111_15 SHARED wall_L2R_111_15.cc)
target_link_libraries(wall_L2R_111_15 ${GAZEBO_LIBRARIES})

add_library(wall_L2R_172_16 SHARED wall_L2R_172_16.cc)
target_link_libraries(wall_L2R_172_16 ${GAZEBO_LIBRARIES})

add_library(wall_L2R_133_17 SHARED wall_L2R_133_17.cc)
target_link_libraries(wall_L2R_133_17 ${GAZEBO_LIBRARIES})

add_library(wall_L2R_110_18 SHARED wall_L2R_110_18.cc)
target_link_libraries(wall_L2R_110_18 ${GAZEBO_LIBRARIES})

add_library(wall_L2R_158_19 SHARED wall_L2R_158_19.cc)
target_link_libraries(wall_L2R_158_19 ${GAZEBO_LIBRARIES})
