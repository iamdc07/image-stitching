def get_path(choice):
    file_list = []
    if choice == 1:
        file_1 = "project_images/Rainier1.png"
        file_2 = "project_images/Rainier2.png"
        file_3 = "project_images/Rainier3.png"
        file_4 = "project_images/Rainier4.png"
        file_5 = "project_images/Rainier5.png"
        file_6 = "project_images/Rainier6.png"

        file_list.append(file_1)
        file_list.append(file_2)
        file_list.append(file_3)
        file_list.append(file_4)
        file_list.append(file_5)
        file_list.append(file_6)

    elif choice == 2:
        file_1 = "project_images/MelakwaLake1.png"
        file_2 = "project_images/MelakwaLake2.png"
        file_3 = "project_images/MelakwaLake3.png"
        file_4 = "project_images/MelakwaLake4.png"
        # file_5 = "s2.png"

        file_list.append(file_1)
        file_list.append(file_2)
        file_list.append(file_3)
        file_list.append(file_4)

    elif choice == 3:
        file_4 = "project_images/pano1_0011.jpg"
        file_3 = "project_images/pano1_0010.jpg"
        file_2 = "project_images/pano1_0009.jpg"
        file_1 = "project_images/pano1_0008.jpg"
        file_5 = "s2.png"

        # file_list.append(file_1)
        # file_list.append(file_2)
        # file_list.append(file_3)
        file_list.append(file_5)
        file_list.append(file_4)

    elif choice == 4:
        file_1 = "project_images/yosemite1.jpg"
        file_2 = "project_images/yosemite2.jpg"

        file_list.append(file_1)
        file_list.append(file_2)

    elif choice == 5:
        file_1 = "project_images/Hanging1.png"
        file_2 = "project_images/Hanging2.png"

        file_list.append(file_1)
        file_list.append(file_2)

    elif choice == 6:
        file_1 = "project_images/img1.ppm"
        file_2 = "project_images/img2.ppm"
        file_3 = "project_images/img4.ppm"

        file_list.append(file_1)
        file_list.append(file_2)
        # file_list.append(file_3)

    elif choice == 7:
        file_1 = "project_images/ND1.png"
        file_2 = "project_images/ND2.png"

        file_list.append(file_1)
        file_list.append(file_2)

    return file_list
