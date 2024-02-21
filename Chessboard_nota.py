import cv2
import numpy as np
import argparse

def map_conners_to_notation(corners, rows, cols):
    """
    This function maps the corners of the chessboard to the chessboard notation
    """
    #Grid notation
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'] 
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
    
    #map the corners to the notation
    grid_coords = {}
    for i in range(rows * cols):
        x, y = corners[i].ravel() #ravel() returns a contiguous flattened array
        grid_coords[(x, y)] = letters[i % cols] + numbers[i // cols] #i % cols gives the column number and i // cols gives the row number

    # Return the notation
    return grid_coords

def print_chessboard(grid_coords, initial_pieces, rows=8, columns=8):
    """Prints a chessboard-like representation with grid coordinates and pieces.

    Args:
        grid_coords: A dictionary mapping corner coordinates to chess notation.
        initial_pieces: A dictionary mapping chess notation to pieces (e.g., {'a1': 'R'}).
        rows: Number of rows on the chessboard (default 8).
        columns: Number of columns on the chessboard (default 8).
    """

    # Build Representation
    board = []
    for row in range(rows, 0, -1):
        board_row = [] 
        for col in range(columns):
            coord = find_coord_by_notation(grid_coords, letters[col] + str(row))
            piece = initial_pieces.get(coord, ' ')  # Get piece from initial_pieces
            board_row.append(f'|{piece}')  # Format cell with piece
        board.append(board_row)

    # Print Board
    print('  ' + '+---+' * columns)
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    for row_index, row in enumerate(board):
        print(f"|{row[0]}|{row[1]}|{row[2]}|{row[3]}|{row[4]}|{row[5]}|{row[6]}|{row[7]}| {rows - row_index}")
        print('  ' + '+---+' * columns)
    print('  ' + ' '.join(letters)) 
    
def find_coord_by_notation(grid_coords, notation):
    """Finds the coordinate of a chessboard corner by notation.

    Args:
        grid_coords: A dictionary mapping corner coordinates to chess 
                     notation (e.g., {(123, 245): 'a8'}).
        notation: The chess notation (e.g., 'a8').

    Returns:
        The coordinate of the corner (e.g., (123, 245)).
    """
    for coord, coord_notation in grid_coords.items():
        if coord_notation == notation:
            return coord
    return None

def find_piece_at_coord(coord):
    """Finds the piece at a given coordinate.

    Args:
        coord: The coordinate of the corner (e.g., (123, 245)).

    Returns:
        The piece at the coordinate (e.g., 'P' for pawn).
    """
    # Replace with your piece detection logic
    return ' '

def crop_square(frame, coord):
    """Crops a square region from the frame based on the given coordinate.

    Assumes coordinates are top-left corners of each grid square. You'll 
    need the size of each square or some way to compute it from your 
    corner detection process.

    Args:
        frame: The image frame.
        coord: A tuple (x, y) representing the top-left corner of the square.

    Returns:
        The cropped square as a new image.
    """

    x, y = coord
    square_size = 50  # Replace with your actual square size 

    return frame[y:y+square_size, x:x+square_size]

def find_coord_with_piece(pieces_dict, piece):
    """Finds the grid coordinate (notation) where a given piece is located.

    Args:
        pieces_dict: A dictionary mapping grid notations to pieces (e.g., {'a1': 'R'}).
        piece: The piece symbol to search for (e.g., 'P', 'K').

    Returns:
        The grid coordinate (e.g., 'a1') if the piece is found, otherwise None.
    """

    for notation, piece_symbol in pieces_dict.items():
        if piece_symbol == piece:
            return notation

    return None  # Piece not found

def detect_piece_in_square(square):
    """Placeholder for piece detection logic.

    Args:
        square: The cropped square image.

    Returns:
        The detected piece symbol (e.g., 'P', 'R', 'K', etc.) or ' ' if empty.
    """

    # Replace this with your actual piece detection logic based on
    # your chosen method (color, template matching, etc.)
    return ' '  # Temporary

def detect_initial_pieces(frame, grid_coords):
    '''Detects the initial pieces on the board and stores them in a dictionary.
       Args: 
         frame: The image frame.
         grid_coords: A dictionary mapping corner coordinates to chess notation.
       returns:
         initial_pieces: A dictionary mapping chess notation to pieces (e.g., {'a1': 'R'}).
    '''
    for coord, notation in grid_coords.items():
        square = crop_square(frame, coord)  # You'll need to implement crop_square
        piece = detect_piece_in_square(square)  # Placeholder - add your logic
        initial_pieces[notation] = piece # Store the piece in a dictionary
        
def detect_movement(frame, grid_coords, initial_pieces):
    '''Detects piece movements on the board and updates the initial_pieces dictionary.
       Args:
         frame: The image frame.
         grid_coords: A dictionary mapping corner coordinates to chess notation.
         initial_pieces: A dictionary mapping chess notation to pieces (e.g., {'a1': 'R'}).
       returns:
         initial_pieces: A dictionary mapping chess notation to pieces (e.g., {'a1': 'R'}).
    '''
    for coord, notation in grid_coords.items():
        current_square = crop_square(frame, coord)
        prev_square = crop_square(prev_frame, coord)  # Store previous frame globally

        diff = cv2.absdiff(current_square, prev_square)
        diff_sum = np.sum(diff)

        if diff_sum > MOVEMENT_THRESHOLD:  # Tune MOVEMENT_THRESHOLD
            initial_pieces[notation] = ' '  # Clear old position
            new_piece = detect_piece_in_square(current_square) 
            if new_piece != ' ':
                new_coord = find_coord_with_piece(initial_pieces, new_piece)
                initial_pieces[new_coord] = new_piece  # Update new position

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Prints a chessboard-like representation with grid coordinates.')
    parser.add_argument('--rows', type=int, default=8, help='Number of rows on the chessboard.')
    parser.add_argument('--columns', type=int, default=8, help='Number of columns on the chessboard.')
    args = parser.parse_args()

    cap = cv2.VideoCapture(0)
    _, prev_frame = cap.read()
    
    initial_pieces = {}  # Stores piece positions (e.g., {'a1': 'R'})
    while True:
        ret, frame = cap.read()
        # Convert the image to grayscale
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (args.columns, args.rows), None)
        # If found, draw corners
        if ret == True:
            # Draw and display the corners
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            cv2.drawChessboardCorners(frame, (args.columns, args.rows), corners, ret)
            grid_coords = map_conners_to_notation(corners, args.rows, args.columns)
            
            detect_initial_pieces(prev_frame, grid_coords)  # Find initial pieces
            detect_movement(frame, grid_coords, initial_pieces)  # Detect piece movements
            print_chessboard(grid_coords, initial_pieces)
         
        cv2.imshow('frame', frame)
        prev_frame = frame.copy()  # Update for the next comparison
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()