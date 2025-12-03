import cv2 as cv
import numpy as np
import os

def show_image(title,image):
    image=cv.resize(image,(0,0),fx=0.3,fy=0.3)
    cv.imshow(title,image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def extract_square(image):
    image_color = cv.resize(image, (768, 1020))
    hsv = cv.cvtColor(image_color, cv.COLOR_BGR2HSV)
    lower = np.array([28, 21, 0])
    upper = np.array([151, 132, 255])
    mask = cv.inRange(hsv, lower, upper)

    kernel = np.ones((5,5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    ys, xs = np.where(mask == 255)
    if len(xs) < 100:
        return None
    
    top_left = min(zip(xs, ys), key = lambda p: p[0] + p[1])
    top_right = max(zip(xs, ys), key = lambda p: p[0] - p[1])
    bottom_left = min(zip(xs, ys), key = lambda p: p[0] - p[1])
    bottom_right = max(zip(xs, ys), key = lambda p: p[0] + p[1])
    src = np.array([top_left, top_right, bottom_left, bottom_right], dtype = np.float32)
    W, H = 1280, 1280
    dst = np.array([[0, 0], [W, 0], [0, H], [W, H]], dtype = np.float32)
    M = cv.getPerspectiveTransform(src, dst)
    result = cv.warpPerspective(image_color, M, (W, H))
    return result


def find_digit(patch):
    color_digits = ["UNU_COLOR4.jpg", "DOI_COLOR.jpg"]
    digit_map = {"UNU_COLOR4.jpg" : 1, "DOI_COLOR.jpg" : 2}
    patch = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
    patch = cv.resize(patch, (80, 80))
    for template in color_digits:
        digit_template = cv.imread('templates/' + template)
        digit_template = cv.resize(digit_template, (80, 80))
        digit_template = cv.cvtColor(digit_template, cv.COLOR_BGR2GRAY)
        for _ in range(4):
            corr = cv.matchTemplate(patch, digit_template, cv.TM_CCOEFF_NORMED)
            corr = np.max(corr)
            if corr > 0.55:
                return digit_map[template]
            digit_template = cv.rotate(digit_template, cv.ROTATE_90_CLOCKWISE)
    return 0


def find_shape(patch):
    shapes = ["5.jpg", "6.jpg", "2.jpg", "3.jpg", "1.jpg", "4.jpg"]
    patch = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
    patch = cv.resize(patch, (80, 80))

    best_corr = 0
    best_shape = 0

    for template in shapes:
        shape_template = cv.imread('templates/' + template)
        shape_template = cv.resize(shape_template, (80, 80))
        shape_template = cv.cvtColor(shape_template, cv.COLOR_BGR2GRAY)
        
        corr = cv.matchTemplate(patch, shape_template, cv.TM_CCOEFF_NORMED)
        corr = np.max(corr)
        if corr > best_corr:
            best_corr = corr
            best_shape = int(template[0])
            best_template = shape_template
    if best_corr > 0.5:
        return best_shape
    return 0

def crop_to_shape(patch, mask):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return patch
    cnt = max(contours, key = cv.contourArea)
    x, y, w, h = cv.boundingRect(cnt)
    crop = patch[y: y + h, x: x + w]
    return crop

def find_color(patch):
    color_ranges = {"R": ((177, 146, 0), (255, 255, 255)), 
                "O": ((0, 144, 156), (24, 255, 255)), 
                "Y": ((24, 132, 58), (59, 255, 255)), 
                "G": ((64, 206, 31), (101, 255, 193)), 
                "B": ((101, 160, 56), (113, 255, 255))}
    patch_hsv = cv.cvtColor(patch, cv.COLOR_BGR2HSV)
    for color, (lower, upper) in color_ranges.items():
        lower_np = np.array(lower)
        upper_np = np.array(upper)
        mask = cv.inRange(patch_hsv, lower_np, upper_np)
        num_color_pixels = cv.countNonZero(mask)
        if num_color_pixels > 150:
            return color
    return "W"

def have_the_same_line(pieces):
    if len(pieces) == 0:
        return False
    for i in range(len(pieces) - 1):
        if pieces[i].split()[0][:-1] != pieces[i + 1].split()[0][:-1]:
            return False
    return True

def have_the_same_column(pieces):
    if len(pieces) == 0:
        return False
    for i in range(len(pieces) - 1):
        if pieces[i].split()[0][-1] != pieces[i + 1].split()[0][-1]:
            return False
    return True

def go_left(piece, pieces):
    coord = piece.split()[0]
    r = int(coord[:-1])
    c = coord[-1]
    col_index = ord(c) - ord('A') + 1
    leftmost = col_index
    for i in range(col_index - 1, 0, -1):
        neighbor_coord = f"{r}{chr(ord('A') + i - 1)}"
        if any(p.startswith(neighbor_coord) for p in pieces):
            leftmost = i
        else:
            break
    return leftmost
            
def go_right(piece, pieces):
    coord = piece.split()[0]
    r = int(coord[:-1])
    c = coord[-1]
    col_index = ord(c) - ord('A') + 1
    rightmost = col_index
    for i in range(col_index + 1, 16):
        neighbor_coord = f"{r}{chr(ord('A') + i - 1)}"
        if any(p.startswith(neighbor_coord) for p in pieces):
            rightmost = i
        else:
            break
    return rightmost

def go_up(piece, pieces):
    coord = piece.split()[0]
    r = int(coord[:-1])
    c = coord[-1]
    row_index = r
    upmost = row_index
    for i in range(row_index - 1, 0, -1):
        neighbor_coord = f"{i}{c}"
        if any(p.startswith(neighbor_coord) for p in pieces):
            upmost = i
        else:
            break
    return upmost

def go_down(piece, pieces):
    coord = piece.split()[0]
    r = int(coord[:-1])
    c = coord[-1]
    row_index = r
    downmost = row_index
    for i in range(row_index + 1, 17):
        neighbor_coord = f"{i}{c}"
        if any(p.startswith(neighbor_coord) for p in pieces):
            downmost = i
        else:
            break
    return downmost

def score_for_round(pieces, added_pieces, bonus_points):
    round_score = 0
    if len(added_pieces) == 1:
        coord = added_pieces[0].split()[0]
        r = int(coord[:-1])
        c = coord[-1]
        up, down, left, right = f"{r - 1}{c}", f"{r + 1}{c}", f"{r}{chr(ord(c) - 1)}", f"{r}{chr(ord(c) + 1)}"
        if any(p.startswith(up) for p in pieces):
            d = r
            u = go_up(up, pieces)
            round_score += d - u + 1
            if d - u + 1 == 6:
                round_score += 6
        elif any(p.startswith(down) for p in pieces):
            u = r
            d = go_down(down, pieces)
            round_score += d - u + 1
            if d - u + 1 == 6:
                round_score += 6
        elif any(p.startswith(left) for p in pieces):
            r = ord(c) - ord('A') + 1
            l = go_left(left, pieces)
            round_score += r - l + 1
            if r - l + 1 == 6:
                round_score += 6
        elif any(p.startswith(right) for p in pieces):
            l = ord(c) - ord('A') + 1
            r = go_right(right, pieces)
            round_score += r - l + 1
            if r - l + 1 == 6:
                round_score += 6
    else:
        if have_the_same_line(added_pieces):
            l = go_left(added_pieces[0], pieces)
            r = go_right(added_pieces[-1], pieces)
            if r - l + 1 != 1:
                round_score += r - l + 1
            if r - l + 1 == 6:
                round_score += 6
            if r - l + 1 > 1:
                for piece in added_pieces:
                    u = go_up(piece, pieces)
                    d = go_down(piece, pieces)
                    if d - u + 1 != 1:
                        round_score += d - u + 1
                    if d - u + 1 == 6:
                        round_score += 6
        elif have_the_same_column(added_pieces):
            u = go_up(added_pieces[0], pieces)
            d = go_down(added_pieces[-1], pieces)
            if d - u + 1 != 1:
                round_score += d - u + 1
            if d - u + 1 == 6:
                round_score += 6
            if d - u + 1 > 1:
                for piece in added_pieces:
                    l = go_left(piece, pieces)
                    r = go_right(piece, pieces)
                    if r - l + 1 != 1:
                        round_score += r - l + 1
                    if r - l + 1 == 6:
                        round_score += 6
    for piece in added_pieces:
        coord = piece.split()[0]
        if coord in bonus_points:
            round_score += bonus_points[coord]
    return round_score

def write_solution(game_id, move_index, added_pieces, score):
    output_dir = '333_Coman_IrinaElena'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = os.path.join(output_dir, f'{game_id}_{move_index:02d}.txt')
    with open(filename, 'w') as f:
        for piece in added_pieces:
            f.write(f"{piece}\n")
        f.write(f"{score}")


lines_horizontal=[]
for i in range(0,1281,80):
    l=[]
    l.append((0,i))
    l.append((1279,i))
    lines_horizontal.append(l)

lines_vertical=[]
for i in range(0,1281,80):
    l=[]
    l.append((i,0))
    l.append((i,1279))
    lines_vertical.append(l)

# HSV mask for pieces detection
lower_contour = np.array([0, 0, 66])
upper_contour = np.array([184, 255, 255])

#HSV mask for digit detection
lower_contour_digit = np.array([0, 147, 160])
upper_contour_digit = np.array([16, 255, 255])

for game in range(1, 6):
    bonus_points = {}
    shapes = []
    pieces = []
    added_pieces = []

    for index in range(21):
        img = cv.imread(f'antrenare/{game}_{index:02d}.jpg')
        image = extract_square(img)
        # show_image('image', image)
        image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        mask_hsv = cv.inRange(image_hsv, lower_contour, upper_contour)
        mask_hsv_digit = cv.inRange(image_hsv, lower_contour_digit, upper_contour_digit)

        for i in range(len(lines_horizontal)-1):
            for j in range(len(lines_vertical)-1):
                y_min = lines_vertical[j][0][0]
                y_max = lines_vertical[j + 1][1][0] 
                x_min = lines_horizontal[i][0][1] 
                x_max = lines_horizontal[i + 1][1][1] 
                patch_full = image[x_min:x_max, y_min:y_max].copy()
                patch_trimmed = image[x_min + 15 : x_max - 15, y_min + 15 : y_max - 15].copy()
                
                if index == 0:
                    mask_patch_digit = mask_hsv_digit[x_min + 15: x_max - 15, y_min + 15: y_max - 15]
                    cropped_patch_trimmed = crop_to_shape(patch_trimmed, mask_patch_digit)
                    exist_digit = find_digit(cropped_patch_trimmed)
                    if exist_digit > 0:
                        coord = f"{i + 1}{chr(ord('A') + j)}"
                        bonus_points[coord] = exist_digit
                    else:
                        mask_patch = mask_hsv[x_min : x_max, y_min : y_max]
                        num_white_pixels = cv.countNonZero(mask_patch)
                        num_black_pixels = mask_patch.size - num_white_pixels
                        if num_white_pixels > 1550 and num_black_pixels > 1550:
                            shapes.append((i + 1, chr(ord('A') + j)))
                            cropped_patch = crop_to_shape(patch_full, mask_patch)
                            exist_shape = find_shape(cropped_patch)
                            if exist_shape > 0:
                                color = find_color(cropped_patch)
                                pieces.append(f"{i + 1}{chr(ord('A') + j)} {exist_shape}{color}")
                else:
                    mask_patch = mask_hsv[x_min : x_max, y_min : y_max]
                    num_white_pixels = cv.countNonZero(mask_patch)
                    num_black_pixels = mask_patch.size - num_white_pixels
                    if num_white_pixels > 1550 and num_black_pixels > 1550:
                        if (i + 1, chr(ord('A') + j)) not in shapes:
                            shapes.append((i + 1, chr(ord('A') + j)))
                            cropped_patch = crop_to_shape(patch_full, mask_patch)
                            exist_shape = find_shape(cropped_patch)
                            if exist_shape > 0:
                                color = find_color(cropped_patch)
                                added_pieces.append(f"{i + 1}{chr(ord('A') + j)} {exist_shape}{color}")

        if index > 0:
            round_score = score_for_round(pieces, added_pieces, bonus_points)
            write_solution(game, index, added_pieces, round_score)
        # if index == 0:
        #     print("Bonus points: ")
        #     print(bonus_points)
        #     print("Shapes positions: ")
        #     print(shapes)
        #     print("Pieces: ")
        #     print(pieces)
        # else:
        # print(str(index) + ": ")
        # print(added_pieces)
        # print("Round score: " + str(round_score))
        pieces.extend(added_pieces)
        added_pieces = []
