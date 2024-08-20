import cv2
import mediapipe as mp 
import streamlit as st


st.title("WELCOME TO FACE DETECTION")
st.write(" DESCRIPTION DU PROJET: Cette petitte application permet de détecter les visage en temps réel ")

if st.button("TESTER"):
    

    # Demander à l'utilisateur d'entrer son nom
    user_name = st.text_input("Veuillez entrer votre nom:")


    #variable qui va détecter les visage
    mp_face_mesh = mp.solutions.face_mesh

    #Variable qui va dessiner les fomres
    mp_drawing = mp.solutions.drawing_utils


    #STYLE DE  LA FORME
    mp_drawing_styles = mp.solutions.drawing_styles

    #ACCES A LA WEBCAME DE L'ORDINATEUR DONC 0 SI C'EST UNE WEBCAM EXTERNE 1
    webcame = cv2.VideoCapture(0)

    while webcame.isOpened():
        success,img = webcame.read()

        #convertir le type de couleur RVB en BGR pour le passer a mediap
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
    # results = mp_face_mesh.FaceMesh(refine_landmarks = True).process(img)
        #SI ON VEUT DETECTER PLUSIEURS VISAGES
        results = mp_face_mesh.FaceMesh(max_num_faces=2,refine_landmarks = True).process(img)

        #RECONVERSION FOR USE OPENCV
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = img.shape

                # Extraire les coordonnées des landmarks
                face_coords = [(int(point.x * w), int(point.y * h)) for point in face_landmarks.landmark]

                        # Utiliser un point central comme référence (entre les yeux)
                forehead_point = face_coords[10]  # Point au-dessus du nez

                        # Vérifier que user_name est une chaîne et non vide
                if user_name :
                    cv2.putText(img,user_name, (forehead_point[0], forehead_point[1] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


                


                #DESSINER LE TESSELATION
                mp_drawing.draw_landmarks(
                    image = img,
                    landmark_list = face_landmarks,
                    connections = mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec = None,
                    connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_tesselation_style()

                )

                #CONTOUR DE VISAGE 

                mp_drawing.draw_landmarks(
                    image = img,
                    landmark_list = face_landmarks,
                    connections = mp_face_mesh.FACEMESH_CONTOURS, 
                    landmark_drawing_spec = None,
                    connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_contours_style()

                )

                #DESSINER LES YEUX
                mp_drawing.draw_landmarks(
                    image = img,
                    landmark_list = face_landmarks,
                    connections = mp_face_mesh.FACEMESH_IRISES, #yeux
                    landmark_drawing_spec = None,
                    connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_iris_connections_style()

                )

                # Afficher le nom de l'utilisateur au-dessus du cadre
                #h, w, _ = img.shape
                #cv2.putText(img, user_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Afficher le nom de l'utilisateur près des contours du visage
                #if user_name:
                   # h, w, _ = img.shape
                # Extraire les coordonnées du contour du visage
                    #face_coords = [(int(point.x * w), int(point.y * h)) for point in face_landmarks.landmark]
                # Utiliser le point de la partie supérieure du visage pour placer le nom
                   # forehead_point = face_coords[10]  # Le point 10 correspond souvent au haut du nez
                    #cv2.putText(img, user_name, (forehead_point[0], forehead_point[1] - 10), 
                               # cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                 # Utiliser un point central comme référence (entre les yeux)
                
                 # Extraire les dimensions de l'image
               

        


        cv2.imshow("koolac",img)

        if cv2.waitKey(5) & 0xFF == ord("q"):
            break 

    webcame.release()
    cv2.destroyAllWindows()