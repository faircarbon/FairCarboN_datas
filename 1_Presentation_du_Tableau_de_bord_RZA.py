import streamlit as st
from PIL import Image

######################################################################################################################
########### TITRE DE L'ONGLET ########################################################################################
######################################################################################################################
st.set_page_config(
    page_title="Tableau de Bord RZA",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "Application de suivi des outils de science ouverte du RZA, développé par Jérôme Dutroncy"}
)

######################################################################################################################
########### COULEURS DES PAGES #######################################################################################
######################################################################################################################
# modification des couleurs utilisées pour le sommaire
st.markdown("""
 <style>
    [data-testid=stSidebar] {
        background-color: rgb(6,51,87,0.2);
    }
    .st-emotion-cache-1dj0hjr {
            color: #a9dba6;
    }
    .st-emotion-cache-1q2d4ya {
            color: #3b979f;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.success("Selectionner une page ci-dessus")

######################################################################################################################
########### NUAGE DE MOTS ############################################################################################
######################################################################################################################
Nuage_mots = Image.open("pages/data/Images/nuage-de-mots.png")
left_co,center_co,last_co = st.columns(3)
with center_co:
    st.sidebar.image(Nuage_mots, width=300)

######################################################################################################################
########### LOGOS ####################################################################################################
######################################################################################################################
col1, col2 = st.columns(2)
logo1 = Image.open("pages/data/Images/logo_CNRS.png")
logo2 = Image.open("pages/data/Images/logo_RZA.png")

with col1:
    st.image(logo1, width=100)
with col2:
    st.image(logo2, width=300)


######################################################################################################################
########### TITRE ET BIENVENUE #######################################################################################
######################################################################################################################
tit = 'Bienvenue sur le Tableau de Bord,ATTENTION NON ACHEVE'
s_tit= f"<p style='font-size:50px;color:rgb(140,140,140)'>{tit}</p>"
st.markdown(s_tit,unsafe_allow_html=True)


guide = st.sidebar.checkbox("Mode d'emploi")

if guide:
    st.title(":grey[Mode d'emploi du site]")

    tab1, tab2, tab3, tab4, tab5 =st.tabs(["Publications HAL","Données RZA","Fiche de métadonnées", "Analyse globale du catalogue", "Data Management" ])

    with tab1:
        publi_hal_guide =  f"""<span style="font-size: 35 px;">
        <ul>
        <li style="font-size:20px";>Une extraction est réalisée sur HAL avec comme requête : le nom de sa Zone Atelier ou l'ensemble des Zones Ateliers. </li>
        <li style="font-size:20px";>La possibilité est laissée à l'utilisateur de choisir l'année du début de sa recherche. </li>
        <li style="font-size:20px";>Si l'on souhaite l'année en cours uniquement, il n'y a rien à faire. </li>
        <li style="font-size:20px";>Sinon, il s'agit de taper l'année de son choix. </li>
        <li style="font-size:20px";>L'extraction sera alors faite jusqu'à la date du jour, sauf si l'on souhaite une année unique (dès lors la case "année unique" est à cocher) </li>
        <li style="font-size:20px";>Il est proposé la possibilité de télécharger le résultat de sa sélection</li>
        </ul></span>"""
        st.markdown(publi_hal_guide, unsafe_allow_html=True)
    with tab2:
        data_guide =  f"""<span style="font-size: 35 px;">
        <ul>
        <li style="font-size:20px";>Une extraction est réalisée sur différents entrepôts (Indores, RDG, Nakala, Zenodo, Dryad, Gbif) ou sur les ressources associées du catalogue (Geonetworks).</li>
        <li style="font-size:20px";>Liste non exhaustive (d'autres peuvent être ajoutés à l'avenir si elle fournissent une API pour le faire.</li>
        <li style="font-size:20px";>Il est proposé de choisir parmi les entrepôts proposés.</li>
        <li style="font-size:20px";>Dès lors, il s'agit de définir sa requête : le nom de sa Zone Atelier ou l'ensemble des Zones Ateliers.</li>
        <li style="font-size:20px";>Il est proposé la possibilité de télécharger le résultat de sa sélection</li>
        </ul></span>"""
        st.markdown(data_guide, unsafe_allow_html=True)
    with tab3:
        fiches_guide =  f"""<span style="font-size: 35 px;">
        <ul>
        <li style="font-size:20px";>Connectée au catalogue InDoRes, cette page propose une lecture Fiche par Fiche des métadonnées enregistrées sur le catalogue.</li>
        </ul>
        <ul style="padding-left: 40px;">
        <li style="font-size:15px";>Attention, la visualisation se fait à partir d'un tableau préenregistré.</li>
        </ul>
        <ul>
        <li style="font-size:20px";>On visualise l'identifieur de la fiche. Un bouton + permet de parcourir les fiches une par une, un bouton R revient au début, un compteur compte les itérations.</li>
        <li style="font-size:20px";>Sur chaque fiche, on retrouve les informations suivantes:</li>
        </ul>
        <ul style="padding-left: 40px;">
        <li style="font-size:15px";>Identification principale: Titre, Fiche parent s'il y en a, Résumé, Purpose (objet), Status, Fréquence de mise à jour, dates multiples</li>
        <li style="font-size:15px";>Métadonnées générales: Date, Langue, Jeu de caractères, type, informations sur les auteurs, Informations sur le standard de la fiche de métadonnées</li>
        <li style="font-size:15px";>Système de référence et limites géographiques: le ou les systèmes renseignés, et les coordonnées cardinales associées aux données</li>
        <li style="font-size:15px";>Mots clés et thésaurus associés</li>
        <li style="font-size:15px";>Contraintes: Limite d'accès ou d'usage, contrainte d'usages ou autres droits associés</li>
        <li style="font-size:15px";>Distribution: Lien URL, Protocole associé, Nom de la ressource, sa description et son format</li>
        <li style="font-size:15px";>Qualité: Niveau, Conformité, Généalogie, et portée</li>
        </ul>
        <ul>
        <li style="font-size:20px";>Sur le côté, est affiché le groupe d'appartenance de la fiche et l'éventuelle mention filtrée au travers des mots clés ou du titre.</li>
        <li style="font-size:20px";>Il est proposé à l'utilisateur de filtrer le réseau de son choix: RZA ou OHM, encochant la case correspondante.</li>
        <li style="font-size:20px";>Dès lors, l'utilisateur pourra sélection le groupe de son choix. </li>
        </ul></span>"""
        st.markdown(fiches_guide, unsafe_allow_html=True)
    with tab4:
        analyse_guide =  f"""<span style="font-size: 35 px;">
         <ul>
        <li style="font-size:20px";>Connectée au catalogue InDoRes, cette page propose une lecture Fiche par Fiche des métadonnées enregistrées sur le catalogue.</li>
        </ul>
        <ul style="padding-left: 40px;">
        <li style="font-size:15px";>Attention, la visualisation se fait à partir d'un tableau préenregistré.</li>
        </ul>
        <ul>
        <li style="font-size:20px";>Cette page permet de visualiser des graphiques généraux suivants:</li>
        </ul>
        <ul style="padding-left: 40px;">
        <li style="font-size:15px";>Répartition des fiches dans les sous-catalogues</li>
        <li style="font-size:15px";>Evolution temporelle des recensements</li>
        <li style="font-size:15px";>Répartition spatiale des données associées au réseau</li>
        <li style="font-size:15px";>Autres champs:</li>
        </ul>
        <ul style="padding-left: 70px;">
        <li style="font-size:15px";>Langues utilisées</li>
        <li style="font-size:15px";>Standards utilisés</li>
        <li style="font-size:15px";>Formats utilisés</li>
        <li style="font-size:15px";>Organisations identifiées</li>
        <li style="font-size:15px";>Contacts identifiés</li>
        <li style="font-size:15px";>Droits et licences associées aux données</li>
        </ul>
        <ul style="padding-left: 40px;">
        <li style="font-size:15px";>Descriptions</li>
        <li style="font-size:15px";>Analyse FAIR</li>
        </ul>
        <ul>
        <li style="font-size:20px";>Il est proposé à l'utilisateur de filtrer le réseau de son choix: RZA ou OHM, encochant la case correspondante.</li>
        <li style="font-size:20px";>Dès lors, l'utilisateur pourra sélection le groupe de son choix. </li>
        <li style="font-size:20px";>Il est possible aussi de choisir de visualiser qu'à partir d'une certaine date (la dernière année par exemple), en ajustant une glissière</li>
        </ul></span>"""
        st.markdown(analyse_guide, unsafe_allow_html=True)
    with tab5:
        data_mana_guide =  f"""<span style="font-size: 35 px;">
        <ul>
        <li style="font-size:20px";>Un bilan est réalisé à partir de différents tableaux extraits précédemment (tableaux complets maintenus par l'administrateur)</li>
        <li style="font-size:20px";>On décompte:</li>
        </ul>
        <ul style="padding-left: 40px;">
        <li style="font-size:15px";>Les publications et selon leur type</li>
        <li style="font-size:15px";>Les données trouvées en accès libre</li>
        <li style="font-size:15px";>Les données citées dans le catalogue => les liens sont régulièrement testés pour savoir s'ils fonctionnent</li>
        </ul>
        <ul>
        <li style="font-size:20px";>Il est aussi réalisé le bilan de champs renseignés ou pas dont un travail de curation sera nécessaire</li>
        </ul>
        <ul style="padding-left: 40px;">
        <li style="font-size:15px";>Conformité de l'identification: nom, affiliation et informations sur l'auteur</li>
        <li style="font-size:15px";>Conformité du format: ouvert ou pas</li>
        <li style="font-size:15px";>Mots clés renseignés</li>
        <li style="font-size:15px";>Thésaurus utilisés</li>
        <li style="font-size:15px";>Droits renseignés</li>
        <li style="font-size:15px";>Généalogie décrite</li>
        </ul>
        <ul>
        <li style="font-size:20px";>Il est proposé à l'utilisateur de filtrer le réseau de son choix: RZA ou OHM, encochant la case correspondante.</li>
        <li style="font-size:20px";>Dès lors, l'utilisateur pourra sélection le groupe de son choix. </li>
        </ul></span>"""
        st.markdown(data_mana_guide, unsafe_allow_html=True)

else:
    
    st.title(':grey[Science Ouverte du RZA]')

    st.subheader("Ce que contient ce site:")

    proposals =  f"""<span style="font-size: 35 px;">
    <ol>
    <li style="font-size:20px";> ICI : Page de présentation et mode d'emploi => Cocher la case à gauche </li>
    <li style="font-size:20px";>Page de recherche des publications sur HAL</li>
    <li style="font-size:20px";>Page de recherche des données déposées / différents entrepôts accessibles</li>
    <li style="font-size:20px";>Visualisation des fiches de métadonnées sur le catalogue InDoRES</li>
    <li style="font-size:20px";>Analyse globale du contenu du catalogue InDoRES</li>
    <li style="font-size:20px";>Une page de bilan et perspectives à traiter</li>
    </ol></span>"""

    st.markdown(proposals, unsafe_allow_html=True)